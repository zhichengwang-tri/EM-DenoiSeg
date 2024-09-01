'''
unified training to see boost
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
from src.utils.show import val_show_dn
from src.utils.utils_training import setup_seed, adjust_learning_rate
from src.loss.loss import MSELoss, BCELoss
from src.loss.loss import WeightedMSE, WeightedBCELoss
from src.utils.seg_mutex import seg_mutex
from src.utils.affinity_ours import multi_offset
from src.data.data_segmentation import relabel
from src.utils.show import show_valid_seg,val_show_dn
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from skimage.metrics import structural_similarity
from src.model.Unet import UNet2D
from src.model.DnCNN import DnCNN
from src.model.RIDNet import RIDNet
from src.model.Unet_multi import Unet_multi as Unet_multi 
from src.model.Unet_multi import Unet_multi_v2 as Unet_multi_v2
from src.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from src.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from src.model.resnet_unet import UNET

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

def build_3model(cfg, writer):
    '''
    使用net替换了原先的seg_denoise
    '''
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    config_vit = CONFIGS_ViT_seg[cfg.MODEL_seg.vit_name]
    config_vit.n_classes = cfg.MODEL_seg.num_classes
    config_vit.n_skip = cfg.MODEL_seg.n_skip
    if cfg.MODEL_seg.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(cfg.TRAIN.img_size / cfg.MODEL_seg.vit_patches_size), int(cfg.TRAIN.img_size / cfg.MODEL_seg.vit_patches_size))
    net = ViT_seg(config_vit, img_size=cfg.TRAIN.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda',0)
    if cfg['MODEL']['MODEL_c_denoise']['model_type'] == 'unet2d':
        model_c_denoise = UNet2D(in_channel=1, out_channel=10, if_sigmoid=True, type = 'denoise').to(device)
    elif cfg['MODEL']['MODEL_c_denoise']['model_type'] == 'dncnn':
        model_c_denoise = DnCNN().to(device)
    elif cfg['MODEL']['MODEL_c_denoise']['model_type'] == 'ridnet':
        model_c_denoise = RIDNet().to(device)
    if cfg['MODEL']['MODEL_denoise']['model_type'] == 'unet-multi':
        model_denoise = Unet_multi(in_channel=cfg.MODEL.MODEL_denoise.input_nc, out_channel=1 ).to(device)
    elif cfg['MODEL']['MODEL_denoise']['model_type'] == 'unet-multi-v2':
        model_denoise = Unet_multi_v2(in_channel=cfg.MODEL.MODEL_denoise.input_nc, out_channel=1 ).to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_denoise = torch.nn.DataParallel(model_denoise)
            model_c_denoise = torch.nn.DataParallel(model_c_denoise)
            net = torch.nn.DataParallel(net)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return net, model_c_denoise, model_denoise

def resume_params(cfg, model_seg, optimizer_seg, model_denoise,optimizer_denoise,model_c_denoise,optimizer_c_denoise,resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.TRAIN.resume_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)
        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model_seg.load_state_dict(checkpoint['model_seg_weights'])
            optimizer_seg.load_state_dict(checkpoint['optimizer_seg_weights'])
            model_denoise.load_state_dict(checkpoint['model_denoise_weights'])
            optimizer_denoise.load_state_dict(checkpoint['optimizer_denoise_weights'])
            model_c_denoise.load_state_dict(checkpoint['model_c_denoise_weights'])
            optimizer_c_denoise.load_state_dict(checkpoint['optimizer_c_denoise_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model_seg, optimizer_seg, model_denoise,optimizer_denoise,model_c_denoise,optimizer_c_denoise, checkpoint['current_iter']
    else:
        return model_seg, optimizer_seg, model_denoise,optimizer_denoise,model_c_denoise,optimizer_c_denoise,0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model_seg,model_c_denoise,model_denoise, ema_model, optimizer_seg,optimizer_c_denoise,optimizer_denoise, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_loss_dn = 0
    sum_loss_seg = 0
    sum_loss_seg_cl = 0
    sum_loss_dn_final = 0
    device = torch.device('cuda:0')
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
    nb_half = cfg.DATA.neighbor // 2

    # criterion_dn = nn.F1Loss()
    if cfg.TRAIN.loss_func_dn == 'MSELoss':
        criterion_dn = nn.MSELoss()
    elif cfg.TRAIN.loss_func_dn == 'Mixloss':
        criterion_dn = MixedLoss()
    else:
        raise AttributeError("NO this criterion")
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
    if cfg.TRAIN.loss_func_cons == 'MSELoss':
        criterion_cons = nn.MSELoss()
    valid_mse = MSELoss()
    best_voi = cfg.TRAIN.best_voi
    fushion_mode = cfg.TRAIN.fushion_mode   
    while iters <= cfg.TRAIN.total_iters:
        # train
        model_seg.train()
        model_denoise.train()
        model_c_denoise.train()
        iters += 1
        t1 = time.time()
        adjust_learning_rate(optimizer_seg, iters, cfg.TRAIN.base_lr, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        adjust_learning_rate(optimizer_denoise, iters, cfg.TRAIN.base_lr, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        adjust_learning_rate(optimizer_c_denoise, iters, cfg.TRAIN.base_lr, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        batch_data = train_provider.next()
        inputs = batch_data['noisy'].cuda()
        target = batch_data['clean'].cuda()
        target_affs = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()  
        coarse_denoise = model_c_denoise(inputs)
        pred_affs = model_seg(coarse_denoise)
        if fushion_mode == 'noisy':
            pred_dn = model_denoise(inputs,pred_affs)
        elif fushion_mode == 'denoised':
            pred_dn = model_denoise(coarse_denoise,pred_affs)
        elif fushion_mode == 'concat':
            pred_dn = model_denoise(torch.concat((coarse_denoise,inputs),dim=1),pred_affs)
        ##############################
        # LOSS
        loss_dn = criterion_dn(coarse_denoise, target) 
        loss_dn_final = criterion_dn(pred_dn,target)  
        loss_seg = criterion_seg(pred_affs, target_affs, weightmap)
        loss = cfg.TRAIN.weight_seg * loss_seg + cfg.TRAIN.weight_dn * loss_dn + cfg.TRAIN.weight_dn_final * loss_dn_final 
        loss.backward()
        ##############################
        optimizer_seg.step()
        optimizer_c_denoise.step()
        optimizer_denoise.step()
        optimizer_seg.zero_grad()
        optimizer_c_denoise.zero_grad()
        optimizer_denoise.zero_grad()
        sum_loss += loss.item()
        sum_loss_dn += loss_dn.item()
        sum_loss_dn_final += loss_dn_final.item()
    
        sum_loss_seg += loss_seg.item()
     
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss=%.6f,loss_dn=%.6f,loss_seg=%.6f,loss_dn_final=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss, sum_loss_dn, sum_loss_seg,sum_loss_dn_final, cfg.TRAIN.base_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60)) 
                writer.add_scalar('loss_total',  sum_loss/cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_dn',  sum_loss_dn/cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_seg',  sum_loss_seg/cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_dn_final',  sum_loss_dn_final/cfg.TRAIN.display_freq, iters)  
            else:
                logging.info('step %d, loss=%.6f,loss_dn=%.6f,loss_seg=%.6f,loss_dn_final=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' \
                            % (iters, sum_loss / cfg.TRAIN.display_freq, sum_loss_dn/cfg.TRAIN.display_freq, sum_loss_seg/ cfg.TRAIN.display_freq, \
                               \
                                    sum_loss_dn_final/cfg.TRAIN.display_freq, cfg.TRAIN.base_lr, sum_time, \
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_dn', sum_loss_dn / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_seg', sum_loss_seg / cfg.TRAIN.display_freq, iters)
                writer.add_scalar('loss_dn_final', sum_loss_dn_final / cfg.TRAIN.display_freq, iters)
                
            f_loss_txt.write('step=%d, loss_dn=%.6f,loss_seg=%.6f,loss_dn_final=%.6f,loss=%.6f' % \
                            (iters, sum_loss_dn/cfg.TRAIN.display_freq,sum_loss_seg/cfg.TRAIN.display_freq, sum_loss_dn_final/cfg.TRAIN.display_freq,sum_loss / cfg.TRAIN.display_freq))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0.0
            sum_loss = 0.0
            sum_loss_seg = 0.0
            sum_loss_dn = 0.0
            sum_loss_dn_final = 0.0
 
        # valid
        if cfg.TRAIN.if_valid:
            save = False
            if iters % cfg.TRAIN.valid_freq == 0 or iters == 1000 and iters>50000:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model_denoise.eval()
                model_seg.eval()
                model_c_denoise.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                losses_valid_dn = []
                losses_valid_seg = []
                all_voi = []
                all_arand = []
                all_psnr = []
                all_ssim = []
                all_psnr_s1 = []
                all_ssim_s1 = []
                for k, batch in tqdm(enumerate(dataloader, 0)):
                    batch_data = batch
                    inputs = batch_data['noisy'].cuda()
                    target = batch_data['clean'].cuda()
                    target_affs = batch_data['affs'].cuda()
                    weightmap = batch_data['wmap'].cuda()
                    pred_affs = torch.zeros(target_affs.shape)
                    coarse_denoise = torch.zeros(inputs.shape)
                    pred_dn = torch.zeros(inputs.shape)
                    with torch.no_grad():
                        for i in range(4):
                            for j in range(4):
                                coarse_denoise_ = model_c_denoise(inputs[:,:,256*i:256*(i+1),256*j:256*(j+1)])
                                pred_affs_ = model_seg(coarse_denoise_)
                                pred_dn_ = model_denoise(coarse_denoise_,pred_affs_)
                                pred_affs[:,:,256*i:256*(i+1),256*j:256*(j+1) ] = pred_affs_ 
                                coarse_denoise[:,:,256*i:256*(i+1),256*j:256*(j+1) ] = coarse_denoise_
                                pred_dn[:,:,256*i:256*(i+1),256*j:256*(j+1) ] = pred_dn_
                    pred_affs = pred_affs.to(device)
                    pred_dn = pred_dn.to(device)
                    coarse_denoise = coarse_denoise.to(device)        
                    loss_dn = criterion_dn(pred_dn, target) 
                    loss_seg = criterion_seg(pred_affs, target_affs, weightmap)
                    tmp_loss = loss_dn 
                    losses_valid.append(tmp_loss.item())
                    losses_valid_dn.append(loss_dn.item())
                    losses_valid_seg.append(loss_seg.item())
                    tmp_mse = valid_mse((pred_dn+1)/2,(target+1)/2)
                    tmp_mse_s1 = valid_mse((coarse_denoise+1)/2,(target+1)/2)
                    out_dn = np.squeeze(pred_dn.data.cpu().numpy()[0,-1])
                    out_dn_s1 = np.squeeze(coarse_denoise.data.cpu().numpy()[0,-1])
                    clean_gt = np.squeeze(target.data.cpu().numpy()[0,-1])
                    noise = np.squeeze(inputs.data.cpu().numpy()[0,-1])
                    out_affs = np.squeeze(pred_affs.data.cpu().numpy())
                    tmp_psnr = 20 * math.log10(1.0 / math.sqrt(tmp_mse))
                    tmp_psnr_s1 = 20 *  math.log10(1.0 / math.sqrt(tmp_mse_s1))
                    tmp_ssim = structural_similarity( (out_dn+1)/2, (clean_gt+1)/2,win_size=7,data_range=1)
                    tmp_ssim_s1 = structural_similarity( (out_dn_s1+1)/2, (clean_gt+1)/2,win_size=7,data_range=1)
                    # evaluate
                    all_psnr.append(tmp_psnr)  
                    all_ssim.append(tmp_ssim)
                    all_psnr_s1.append(tmp_psnr_s1)  
                    all_ssim_s1.append(tmp_ssim_s1)
                    # post-processing
                    gt_ins = np.squeeze(batch_data['seg'].numpy()).astype(np.uint8)
                    gt_mask = gt_ins.copy()
                    gt_mask[gt_mask != 0] = 1
                    pred_seg = seg_mutex(out_affs, offsets=offsets, strides=list(cfg.DATA.strides), mask=gt_mask).astype(np.uint16)
                    pred_seg = relabel(pred_seg)
                    pred_seg = pred_seg.astype(np.uint16)
                    gt_ins = gt_ins.astype(np.uint16)
                    # evaluate
                    arand = adapted_rand_ref(gt_ins, pred_seg, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(gt_ins, pred_seg, ignore_labels=(0))
                    voi_sum = voi_split + voi_merge
                    all_voi.append(voi_sum)
                    all_arand.append(arand)                   
                    if k == 0:
                        affs_gt = batch_data['affs'].numpy()[0,-1]  
                    if iters % cfg.TRAIN.save_freq == 0:
                        val_show_dn(iters,out_dn, clean_gt, noise, cfg.valid_path,k)
                        show_valid_seg(iters,noise,pred_seg,gt_ins,cfg.valid_path,k)
                epoch_loss = sum(losses_valid) / len(losses_valid)
                epoch_loss_dn = sum(losses_valid_dn) / len(losses_valid_dn)
                epoch_loss_seg = sum(losses_valid_seg) / len(losses_valid_seg)
                mean_psnr = sum(all_psnr) / len(all_psnr) 
                mean_ssim = sum(all_ssim) / len(all_ssim)  
                mean_psnr_s1 = sum(all_psnr_s1) / len(all_psnr_s1) 
                mean_ssim_s1 = sum(all_ssim_s1) / len(all_ssim_s1)  
                mean_voi = sum(all_voi) / len(all_voi)
                mean_arand = sum(all_arand) / len(all_arand)
                if mean_voi < best_voi:
                    best_voi = mean_voi
                    save = True
                print('model-%d, valid-loss=%.6f VOI=%.6f, ARAND=%.6f,PSNR=%.6f,SSIM=%.6f' % \
                    (iters, epoch_loss, mean_voi, mean_arand,mean_psnr,mean_ssim), flush=True) 
                writer.add_scalar('valid/VOI', mean_voi, iters)
                writer.add_scalar('valid/ARAND', mean_arand, iters)
                writer.add_scalar('valid/PSNR', mean_psnr, iters)
                writer.add_scalar('valid/SSIM', mean_ssim, iters)
                writer.add_scalar('valid/PSNR_s1', mean_psnr_s1, iters)
                writer.add_scalar('valid/SSIM_S1', mean_ssim_s1, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f,valid-loss-dn =%.6f, valid-loss-seg =%.6f, VOI=%.6f, ARAND=%.6f, PSNR=%.6f, SSIM=%.6f,PSNR_S1=%.6f, SSIM_S1=%.6f' % \
                    (iters, epoch_loss, epoch_loss_dn,epoch_loss_seg, mean_voi, mean_arand, mean_psnr, mean_ssim, mean_psnr_s1, mean_ssim_s1))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

            # save
            if iters%cfg.TRAIN.valid_freq==0 and iters>20000 and save:
                states = {'current_iter': iters, 'valid_result': None,
                        'model_denoise_weights': model_denoise.state_dict(),
                        'model_c_denoise_weights': model_c_denoise.state_dict(),
                        'model_seg_weights': model_seg.state_dict(),
                        'optimizer_seg_weights': optimizer_seg.state_dict(),
                        'optimizer_denoise_weights': optimizer_denoise.state_dict(),
                        'optimizer_c_denoise_weights': optimizer_c_denoise.state_dict(),
                        }
                torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
                print('***************save model, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='unified_v2_c_trans_ridnet', help='path to config file')
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
        model_seg,model_c_denoise,model_denoise = build_3model(cfg, writer)
        ema_model = None
        if cfg.TRAIN.opt_type == 'sgd':
            optimizer_seg = optim.SGD( filter(lambda p: p.requires_grad,model_seg.parameters() ), lr=cfg.TRAIN.base_lr, momentum=0.9, weight_decay=0.0001)
            optimizer_c_denoise = optim.SGD( filter(lambda p: p.requires_grad,model_c_denoise.parameters() ), lr=cfg.TRAIN.base_lr, momentum=0.9, weight_decay=0.0001)
            optimizer_denoise = optim.SGD( filter(lambda p: p.requires_grad,model_denoise.parameters() ), lr=cfg.TRAIN.base_lr, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer_seg = torch.optim.Adam(filter(lambda p: p.requires_grad,model_seg.parameters() ), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)
            optimizer_denoise = torch.optim.Adam(filter(lambda p: p.requires_grad,model_denoise.parameters() ), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)
            optimizer_c_denoise = torch.optim.Adam(filter(lambda p: p.requires_grad,model_c_denoise.parameters() ), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    eps=0.01, weight_decay=1e-6, amsgrad=True)
            
        model_seg, optimizer_seg,model_denoise,optimizer_denoise,model_c_denoise,optimizer_c_denoise,init_iters = resume_params(cfg, model_seg, optimizer_seg, model_denoise,optimizer_denoise,model_c_denoise,optimizer_c_denoise,cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model_seg,model_c_denoise, model_denoise, ema_model, optimizer_seg,optimizer_c_denoise,optimizer_denoise, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')