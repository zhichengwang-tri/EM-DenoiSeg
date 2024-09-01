'''
test-dn-v2
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

from src.utils.utils_training import setup_seed,adjust_learning_rate
from src.loss.loss import WeightedMSE, WeightedBCELoss
from src.loss.loss import MSELoss, BCELoss
from src.loss.loss_fa import FALoss
from src.utils.seg_mutex import seg_mutex
from src.utils.affinity_ours import multi_offset
from src.data.data_segmentation import relabel
from src.model.Unet import UNet2D
from src.model.DnCNN import DnCNN
from src.model.RIDNet import RIDNet
from src.model.resnet_unet import UNET
from src.utils.show import show_valid_seg,val_show_dn
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings("ignore")
from src.utils.show import draw_fragments_2d
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
    device = torch.device('cuda:0') 
    if cfg['MODEL']['MODEL_denoise']['model_type'] == 'unet':
        model_denoise = UNet2D(in_channel=1, out_channel=10, if_sigmoid=True, type = 'denoise').to(device)
    elif cfg['MODEL']['MODEL_denoise']['model_type'] == 'dncnn':
        model_denoise = DnCNN().to(device)
    elif cfg['MODEL']['MODEL_denoise']['model_type'] == 'ridnet':
        model_denoise = RIDNet().to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_denoise = torch.nn.DataParallel(model_denoise,device_ids = [0])
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_denoise


def resume_params_dn(cfg, model, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.TRAIN.resume_path_dn, 'model-%06d.ckpt' % cfg.TRAIN.model_id_dn)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_weights']
        partial_dict = {}
        for k,v in state_dict.items():
                partial_dict[k] = v
        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            model.load_state_dict(partial_dict,strict = False)
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        return model, checkpoint['current_iter']
    else:
        return model, 0
def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, valid_provider, model_dn, writer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    offsets = multi_offset(list(cfg.DATA.shifts), neighbor=cfg.DATA.neighbor)
    valid_mse = MSELoss()
    model_dn.eval()
    dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                    shuffle=False, drop_last=False, pin_memory=True)

    all_psnr = []
    all_ssim = []
    for k, batch in tqdm(enumerate(dataloader, 0)):
        batch_data = batch
        inputs = batch_data['noisy'].cuda()
        target = batch_data['clean'].cuda()
        target_affs = batch_data['affs'].cuda()
        weightmap = batch_data['wmap'].cuda()
        denoised = torch.zeros((1,1,1024,1024))
        with torch.no_grad():
            for i in range(4):
                for j in range(4):
                    denoised_ = model_dn(inputs[:,:,256*i:256*(i+1),256*j:256*(j+1)])
                    denoised[:,:,256*i:256*(i+1),256*j:256*(j+1) ] = denoised_      
        denoised = denoised.to(device)     
        denoised = torch.clamp(denoised,-1,1)
        tmp_mse = valid_mse((denoised+1)/2,(target+1)/2)
        denoised = np.squeeze(denoised.data.cpu().numpy()[0,-1])
        clean_gt = np.squeeze(target.data.cpu().numpy()[0,-1])
        tmp_psnr = 20 * math.log10(1.0 / math.sqrt(tmp_mse))
        tmp_ssim = structural_similarity( (denoised+1)/2, (clean_gt+1)/2,win_size=7,data_range=1)
        # evaluate
        all_psnr.append(tmp_psnr) 
        all_ssim.append(tmp_ssim)
        noise = np.squeeze(inputs.data.cpu().numpy()[0,-1]) 
        mean_psnr = sum(all_psnr) / len(all_psnr)
        mean_ssim = sum(all_ssim) / len(all_ssim)
        '''save picturesc dn'''
        os.makedirs(os.path.join(cfg.valid_path, 'dn'), exist_ok=True)
        denoised = denoised[:,:,np.newaxis]
        denoised = np.repeat(denoised, 3, 2)
        denoised = ((denoised + 1 )/2 * 255).astype(np.uint8)
        Image.fromarray(denoised).save(os.path.join(cfg.valid_path, 'dn','{0}.png'.format(k))) 
      
    f_valid_txt.write('PSNR=%.6f,SSIM=%.6f' % \
        (mean_psnr,mean_ssim))
    f_valid_txt.write('\n')
    f_valid_txt.flush()
    torch.cuda.empty_cache()

    # save
  
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='test_dn_ac4', help='path to config file')
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
        _, valid_provider = load_dataset(cfg)
        model_dn = build_model(cfg, writer)
        model_dn, init_iters = resume_params_dn(cfg, model_dn, cfg.TRAIN.resume)
        loop(cfg, valid_provider, model_dn, writer)
        writer.close()
    else:
        pass
    print('***Done***')