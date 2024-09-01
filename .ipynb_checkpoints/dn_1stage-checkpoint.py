'''
denoise
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import yaml
import time
import cv2
import h5py
import random
import math
import logging
import argparse
import numpy as np
# monai sliding window inference
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.loss.loss_denoise import MixedLoss
from src.data.data_provider import Provider, Validation
from src.model.RIDNet import RIDNet
from src.model.DnCNN import DnCNN
from src.utils.utils_training import setup_seed,adjust_learning_rate
from src.loss.loss import WeightedMSE, WeightedBCELoss
from src.loss.loss import MSELoss, BCELoss
from src.utils.seg_mutex import seg_mutex
from src.utils.affinity_ours import multi_offset
from src.data.data_segmentation import relabel
from src.model.Unet import UNet2D
from src.model.resnet_unet import UNET
from src.utils.show import show_valid_seg,val_show_dn
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = prefix 
    else:
        model_name = prefix 
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, cfg.NAME, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, cfg.NAME, model_name)
    cfg.record_path = os.path.join(cfg.TRAIN.record_path, cfg.NAME, model_name)
    cfg.valid_path = os.path.join(cfg.TRAIN.valid_path,  cfg.NAME, model_name)

    if not os.path.exists(cfg.cache_path):
        os.makedirs(cfg.cache_path)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    if not os.path.exists(cfg.record_path):
        os.makedirs(cfg.record_path)
    if not os.path.exists(cfg.valid_path):
        os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Validation(root_path=cfg.DATA.root_path,
                                data_name=cfg.DATA.data_name,
                                num_train=cfg.DATA.num_train,
                                num_valid=cfg.DATA.num_valid,
                                num_test=cfg.DATA.num_test,
                                crop_size=list(cfg.DATA.crop_size),
                                padding=0,
                               )
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda',0)
    if cfg['MODEL']['MODEL_denoise']['model_type'] == 'unet':
        model = UNet2D(in_channel=1, out_channel=10, if_sigmoid=True, type = 'denoise').to(device)
    elif cfg['MODEL']['MODEL_denoise']['model_type'] == 'dncnn':
        model = DnCNN().to(device)
    elif cfg['MODEL']['MODEL_denoise']['model_type'] == 'ridnet':
        model = RIDNet().to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = torch.nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr

def loop(cfg, train_provider, valid_provider, model_uni, ema_model, optimizer_uni, iters, writer):
    iters = 0
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0 
    sum_loss_dn = 0
    # sum_loss_embedding = 0.0
    # sum_loss_mask = 0.0
    device = torch.device('cuda:0')
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
    nb_half = cfg.DATA.neighbor // 2

    if cfg.TRAIN.loss_func_seg== 'MSELoss':
        criterion_seg = MSELoss()
    elif cfg.TRAIN.loss_func_seg == 'BCELoss':
        criterion_seg = BCELoss()
    elif cfg.TRAIN.loss_func_seg == 'WeightedMSELoss':
        criterion_seg = WeightedMSE()
    elif cfg.TRAIN.loss_func_seg == 'WeightedBCELoss':
        criterion_seg = WeightedBCELoss()
    else:
        raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_func_dn == 'MSELoss':
        criterion_dn = nn.MSELoss()
    elif cfg.TRAIN.loss_func_dn == 'Mixloss':
        criterion_dn = MixedLoss()
    else:
        raise AttributeError("NO this criterion")
    valid_mse = MSELoss()
    best_psnr = cfg.TRAIN.best_psnr
    while iters <= cfg.TRAIN.total_iters:
        # train
        model_uni.train()
        iters += 1
        t1 = time.time()
        adjust_learning_rate(optimizer_uni, iters, cfg.TRAIN.base_lr, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        batch_data = train_provider.next()
        inputs = batch_data['noisy'].cuda()
        target = batch_data['clean'].cuda()
        target_affs = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()  
        denoised= model_uni(inputs)
        ##############################
        # LOSS
        loss_dn = criterion_dn(denoised,target) 
        loss = loss_dn
        loss.backward()
        ##############################
        optimizer_uni.step()
        optimizer_uni.zero_grad()
        sum_loss += loss.item()
        sum_loss_dn += loss_dn.item()
        sum_time += time.time() - t1
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss=%.6f,loss_dn=%.6f(wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss, sum_loss_dn, cfg.TRAIN.base_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60)) 
                writer.add_scalar('loss_total',  sum_loss/cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_dn',  sum_loss_dn/cfg.TRAIN.display_freq, iters)
            
            else:
                logging.info('step %d, loss=%.6f,loss_dn=%.6f(wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                            % (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_dn/cfg.TRAIN.display_freq, cfg.TRAIN.base_lr, sum_time, \
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_dn', sum_loss_dn / cfg.TRAIN.display_freq, iters)
              
                
            f_loss_txt.write('step=%d, loss_dn=%.6f,loss=%.6f' % \
                            (iters, sum_loss_dn/cfg.TRAIN.display_freq,sum_loss / cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0.0
            sum_loss = 0.0
            sum_loss_dn = 0.0
             # valid
        
        if cfg.TRAIN.if_valid:
            save = False
            if iters % cfg.TRAIN.valid_freq == 0 or iters == 1000:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model_uni.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                losses_valid_dn = []
                all_psnr = []
                all_ssim = []
               
                for k, batch in tqdm(enumerate(dataloader, 0)):
                    batch_data = batch
                    inputs = batch_data['noisy'].cuda()
                    target = batch_data['clean'].cuda()
                    denoised = torch.zeros((1,1,1024,1024))
                    with torch.no_grad():
                        for i in range(4):
                            for j in range(4):
                                denoised_ = model_uni(inputs[:,:,256*i:256*(i+1),256*j:256*(j+1)])
                                denoised[:,:,256*i:256*(i+1),256*j:256*(j+1) ] = denoised_ 
                    denoised = denoised.to(device)       
                    loss_dn = criterion_dn (denoised,target)
                    tmp_loss = loss_dn 
                    losses_valid.append(tmp_loss.item())
                    losses_valid_dn.append(loss_dn.item())
                    tmp_mse = valid_mse((denoised+1)/2,(target+1)/2)
                    denoised = np.squeeze(denoised.data.cpu().numpy()[0,-1])
                    clean_gt = np.squeeze(target.data.cpu().numpy()[0,-1])
                    noise = np.squeeze(inputs.data.cpu().numpy()[0,-1]) 
                    tmp_psnr = 20 * math.log10(1.0 / math.sqrt(tmp_mse))
                    tmp_ssim = structural_similarity( (denoised+1)/2, (clean_gt+1)/2,win_size=7,data_range=1)
                    # evaluate
                    all_psnr.append(tmp_psnr) 
                    all_ssim.append(tmp_ssim)
                    mean_psnr = sum(all_psnr) / len(all_psnr)
                    mean_ssim = sum(all_ssim) / len(all_ssim)
                    # post-processing
                    if mean_psnr > best_psnr:
                        best_psnr = mean_psnr
                        save = True
                    if iters % cfg.TRAIN.save_freq == 0:
                        val_show_dn(iters, denoised, clean_gt, noise, cfg.valid_path,k)
                epoch_loss = sum(losses_valid) / len(losses_valid)
                epoch_loss_dn = sum(losses_valid_dn) / len(losses_valid_dn)
               
                writer.add_scalar('valid/PSNR', mean_psnr, iters)
                writer.add_scalar('valid/SSIM', mean_ssim, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f,valid-loss-dn =%.6f,PSNR=%.6f,SSIM=%.6f' % \
                    (iters, epoch_loss, epoch_loss_dn,mean_psnr,mean_ssim))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

            # save
            if iters%cfg.TRAIN.valid_freq==0  and iters>200000 and save:
                states = {'current_iter': iters, 'valid_result': None,
                        'model_weights': model_uni.state_dict(),
                        'optimizer_weights':optimizer_uni.state_dict()}
                torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
                print('***************save model, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='dn_1stage_c_ridnet', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()
    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)
    config_path = os.path.join('./conf', cfg_file)
    with open(config_path, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model_uni = build_model(cfg, writer)
        ema_model = None
        if cfg.TRAIN.opt_type == 'sgd':
            optimizer_uni = optim.SGD(filter(lambda p: p.requires_grad,model_uni.parameters() ), lr=cfg.TRAIN.base_lr, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer_uni = torch.optim.Adam(filter(lambda p: p.requires_grad,model_uni.parameters() ), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)         
        loop(cfg, train_provider, valid_provider, model_uni, ema_model, optimizer_uni, 0 , writer)
        writer.close()
    else:
        pass
    print('***Done***')