'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2022-03-20 15:15:58
'''
import os
import sys
import time
import h5py
import torch
import random
import numpy as np
from PIL import Image
import os.path as osp
from torch.utils import data
from torch.utils.data import DataLoader
from ..utils.pre_processing import normalization2, approximate_image, cropping
from ..data.data_aug import aug_img_lab
from src.utils.affinity_ours import multi_offset, gen_affs_ours
from src.data.data_segmentation import weight_binary_ratio
from scipy.ndimage import gaussian_filter
from torchvision.transforms import (Compose, Lambda, ToTensor)
import copy
def generate_mask(input,ratio = 0.1, size_window = [4,4], size_data = [256,256] ):
    num_sample = int(size_data[0] * size_data[1] * (1 - ratio))
    mask = np.ones(size_data)
    output = input
    idy_msk = np.random.randint(0, size_data[0], num_sample)
    idx_msk = np.random.randint(0, size_data[1], num_sample)
    idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
    idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)
    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh
    idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
    idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]
    id_msk = (idy_msk, idx_msk)
    id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)
    output[id_msk] = input[id_msk_neigh]
    mask[id_msk] = 0.0
    return output, mask
            
class TrainingDataSet(data.Dataset):
    def __init__(self, root_path='/braindat/lab/chenyd/DATASET',
                    data_name='CREMIC',
                    num_train=75,
                    crop_size=[224,224],
                    separate_weight=True,
                    shifts=[1,3,5,9,15],
                    neighbor=4,
                   ):
        self.root_path = root_path  # 
        self.data_name = data_name  # 
        self.crop_size = crop_size  # [256, 256]
        self.separate_weight = separate_weight
        self.offsets = multi_offset(list(shifts), neighbor=neighbor)
        self.crop_size_pad = [512, 512]
        self.num_train = num_train
        if self.data_name == 'CREMIA':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiA/cremiA_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiA/', 'cremiA_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiA/noisy_gp.npy')
        elif  self.data_name == 'CREMIC':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiC/cremiC_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiC/', 'cremiC_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiC/noisy_gp.npy')
        elif self.data_name == 'CREMIB':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiB/cremiB_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiB/', 'cremiB_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiB/noisy_gp.npy')    
        elif self.data_name == 'AC3/4':
            raw_path = osp.join('/data/ZCWANG007/cremi/AC3/AC3_inputs.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/AC3/', 'AC3_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/AC3/noisy_gp.npy')
        elif self.data_name == 'AC3/4-film':
            raw_path = osp.join('/data/ZCWANG007/cremi/AC3/AC3_inputs.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/AC3/', 'AC3_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/AC3/noisy_film.npy')
        print('Load raw:', raw_path)
        f = h5py.File(raw_path, 'r')
        self.total_raw = f['main'][:]
        f.close()
        f = h5py.File(label_path, 'r')
        self.total_label = f['main'][:]
        f.close()
        self.total_noisy = np.load(noisy_path)
        self.training_raw = self.total_raw[:self.num_train]
        self.training_noisy = self.total_noisy[:self.num_train]
        # 仅能使用和后续finetune时相同的label数量
        self.training_label = self.total_label[:self.num_train]
        print('Training shape:', self.training_raw.shape)
        self.num_training = self.training_raw.shape[0]
        self.transforms = Compose(
            [
                ToTensor(),
                Lambda(lambda t : (t*2)-1),
            ]
        )

    def __len__(self):
        return int(sys.maxsize)
        # return 10
 
    
    def __getitem__(self, idx):
        
        k = random.randint(0, self.num_training - 1)
        img = self.training_raw[k] 
        size = img.shape
        noisy = self.training_noisy[k].astype(np.uint8)
        img = self.transforms(img)
        noisy = self.transforms(noisy)
        label = self.training_label[k] 
        # # data augmentation
        # img, label = aug_img_lab(img, label, self.crop_size)
        # img = normalization2(img.astype(np.float32), max=1, min=0)
        y_loc = random.randint(0, size[0] - self.crop_size[0] - 1)
        x_loc = random.randint(0, size[1] - self.crop_size[1] - 1)
        img = img[:,y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        noisy = noisy[:,y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        label = label[y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        label = label.astype(np.uint64)
        lb_affs, affs_mask = gen_affs_ours(label, offsets=self.offsets, ignore=False, padding=True)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(len(self.offsets)):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)       
        lb_affs = torch.from_numpy(lb_affs).float()
        weightmap = torch.from_numpy(weightmap).float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        return {'clean': img,
                'noisy': noisy,
                'affs': lb_affs,
                'wmap': weightmap,
                'seg': label,
                } 
        
        

class TrainingDataSet_jdas(data.Dataset):
    def __init__(self, root_path='/braindat/lab/chenyd/DATASET',
                    data_name='CREMI',
                    num_train=75,
                    crop_size=[256, 256],
                    separate_weight=True,
                    shifts=[1,3,5,9,15],
                    neighbor=4,
                    jdas = False,
                    label_percentage= '1' # 1 8 15 38 75
                   ):
        self.root_path = root_path  # 
        self.data_name = data_name  # 
        self.crop_size = crop_size  # [256, 256]
        self.separate_weight = separate_weight
        self.offsets = multi_offset(list(shifts), neighbor=neighbor)
        self.crop_size_pad = [512, 512]
        self.num_train = num_train
        self.label_percentage = label_percentage
        self.jdas = jdas
        if self.data_name == 'CREMIC':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiC/cremiC_inputs_interp.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiC/noisy_gp.npy')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiC/', 'cremiC_labels.h5')
        elif self.data_name == 'CREMIA':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiA/cremiA_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiA/', 'cremiA_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiA/noisy_gp.npy')
        elif self.data_name == 'CREMIB':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiB/cremiB_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiB/', 'cremiB_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiB/noisy_gp.npy')
        elif self.data_name == 'AC3/4':
            raw_path = osp.join('/data/ZCWANG007/cremi/AC3/AC3_inputs.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/AC3/', 'AC3_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/AC3/noisy_gp.npy')
        
        print('Load raw:', raw_path)
        f = h5py.File(raw_path, 'r')
        self.total_raw = f['main'][:]
        f.close()
        f = h5py.File(label_path, 'r')
        self.total_label = f['main'][:]
        f.close()
        self.total_noisy = np.load(noisy_path)
        self.training_raw = self.total_raw[:self.num_train]
        self.training_noisy = self.total_noisy[:self.num_train]
        # 仅能使用和后续finetune时相同的label数量
        self.training_label = self.total_label[:self.num_train]
        print('Training shape:', self.training_raw.shape)
        self.num_training = self.training_raw.shape[0]
        self.transforms = Compose(
            [
                ToTensor(),
                Lambda(lambda t : (t*2)-1),
            ]
        )

    def __len__(self):
        return int(sys.maxsize)
        # return 10
    
    def __getitem__(self, idx):
        
        k = random.randint(0, self.num_training - 1)
        if self.label_percentage == '1':
            if k == 0:
                label_pretrain = True
            else:
                label_pretrain = False
        elif self.label_percentage == '8':
            if k // 8 == 0:
                label_pretrain = True
            else:
                label_pretrain = False
        elif self.label_percentage == '15':
            if k // 15 == 0:
                label_pretrain = True
            else:
                label_pretrain = False
        elif self.label_percentage == '38':
            if k  // 38 == 0:
                label_pretrain = True
            else:
                label_pretrain = False
        elif self.label_percentage == '75':
            if k // 75 == 0:
                label_pretrain = True
            else:
                label_pretrain = False
        img = self.training_raw[k] 
        size = img.shape
        noisy = self.training_noisy[k].astype(np.uint8)
        img = self.transforms(img)
        noisy = self.transforms(noisy)
        label = self.training_label[k]
        y_loc = random.randint(0, size[0] - self.crop_size[0] - 1)
        x_loc = random.randint(0, size[1] - self.crop_size[1] - 1)
        img = img[:,y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        noisy = noisy[:,y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        label = label[y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        label = label.astype(np.uint64)
        lb_affs, affs_mask = gen_affs_ours(label, offsets=self.offsets, ignore=False, padding=True)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(len(self.offsets)):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)     
        lb_affs = torch.from_numpy(lb_affs).float()
        weightmap = torch.from_numpy(weightmap).float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        return {'clean': img,
                'noisy': noisy,
                'affs': lb_affs,
                'wmap': weightmap,
                'seg': label,
                'label_pretrain': label_pretrain} 
        
              
class Validation(data.Dataset):
    def __init__(self, root_path='../data',
                    data_name='CREMIC',
                    mode='valid',
                    num_train=75,
                    num_valid=50,
                    num_test=50,
                    crop_size=[256, 256],
                    padding=0,
                    shifts=[1,3,5,9,15],
                    neighbor=4,
                    separate_weight= True,
                    ):
        self.padding = padding
        self.root_path = root_path  # ../data
        self.data_name = data_name  # ISBI2012, ISBI2013, CREMIA
        self.crop_size = crop_size  # [256, 256]
        self.separate_weight = separate_weight       
        if self.data_name == 'CREMIC':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiC/cremiC_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiC/', 'cremiC_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiC/noisy_gp.npy')
            affinity_path = osp.join('/data/ZCWANG007/cremi/cremiC/affinity_c.npy')
        elif self.data_name == 'CREMIA':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiA/cremiA_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiA/', 'cremiA_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiA/noisy_gp.npy')
            affinity_path = osp.join('/data/ZCWANG007/cremi/cremiA/affinity_a.npy')
        elif self.data_name == 'CREMIB':
            raw_path = osp.join('/data/ZCWANG007/cremi/cremiB/cremiB_inputs_interp.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/cremiB/', 'cremiB_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/cremiB/noisy_gp.npy')
            affinity_path = osp.join('/data/ZCWANG007/cremi/cremiB/affinity_b.npy')
        elif self.data_name == 'AC3/4':
            raw_path = osp.join('/data/ZCWANG007/cremi/AC4/AC4_inputs.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/AC4/', 'AC4_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/AC4/noisy_gp.npy')
            affinity_path = osp.join('/data/ZCWANG007/cremi/AC4/affinity_ac4.npy')
        elif self.data_name == 'AC3/4-film':
            raw_path = osp.join('/data/ZCWANG007/cremi/AC4/AC4_inputs.h5')
            label_path = osp.join('/data/ZCWANG007/cremi/AC4/', 'AC4_labels.h5')
            noisy_path = osp.join('/data/ZCWANG007/cremi/AC4/noisy_film.npy')
            affinity_path = osp.join('/data/ZCWANG007/cremi/AC4/affinity_ac4.npy')
        print('Load raw:', raw_path)
        f = h5py.File(raw_path, 'r')
        self.total_raw = f['main'][:]
        f.close()
        self.total_noisy = np.load(noisy_path)
        self.total_aff = np.load(affinity_path)
        f = h5py.File(label_path, 'r')
        self.total_label = f['main'][:]
        f.close()
        self.valid_raw = self.total_raw[-num_valid:]
        self.valid_noisy = self.total_noisy[-num_valid:]
        self.valid_label = self.total_label[-num_valid:]
        self.valid_aff = self.total_aff[-num_valid:]
        self.offsets = multi_offset(list(shifts), neighbor=neighbor)
        self.transforms = Compose(
            [
                ToTensor(),
                Lambda(lambda t : (t*2)-1),
            ]
        )

    def __len__(self):
        return self.valid_raw.shape[0]
    
    def __getitem__(self, k):
        
        img = self.valid_raw[k]
        noisy = self.valid_noisy[k].astype(np.uint8)
        img = self.transforms(img)
        noisy = self.transforms(noisy)
        lb = self.valid_label[k]
        label = lb.astype(np.uint64)
        lb_affs, affs_mask = gen_affs_ours(label, offsets=self.offsets, ignore=False, padding=True)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(len(self.offsets)):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)
        lb_affs = torch.from_numpy(lb_affs).float()
        weightmap = torch.from_numpy(weightmap).float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        return {'clean': img,
                'noisy': noisy,
                'affs': lb_affs,
                'wmap': weightmap,
                'seg': label}


class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = TrainingDataSet(root_path=cfg.DATA.root_path,
                                        data_name=cfg.DATA.data_name,
                                        num_train=cfg.DATA.num_train,
                                        crop_size=list(cfg.DATA.crop_size),
                                        separate_weight=cfg.DATA.separate_weight,
                                        shifts=list(cfg.DATA.shifts),
                                        neighbor=cfg.DATA.neighbor,
                                       )
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        # return self.data.num_per_epoch
        return int(sys.maxsize)

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                            shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch['noisy'] = batch['noisy'].cuda()   
                batch['clean'] = batch['clean'].cuda()
                batch['affs'] = batch['affs'].cuda()
                batch['wmap'] = batch['wmap'].cuda()
                batch['seg'] = batch['seg'].cuda()
               
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch['clean'] = batch['clean'].cuda()
                batch['noisy'] = batch['noisy'].cuda()
                batch['affs'] = batch['affs'].cuda()
                batch['wmap'] = batch['wmap'].cuda()
                batch['seg'] = batch['seg'].cuda()
               
            return batch

class Provider_jdas(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = TrainingDataSet_jdas(root_path=cfg.DATA.root_path,
                                        data_name=cfg.DATA.data_name,
                                        num_train=cfg.DATA.num_train,
                                        crop_size=list(cfg.DATA.crop_size),
                                        separate_weight=cfg.DATA.separate_weight,
                                        shifts=list(cfg.DATA.shifts),
                                        neighbor=cfg.DATA.neighbor,
                                        jdas= True,
                                        label_percentage=cfg.TRAIN.label_percentage)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        # return self.data.num_per_epoch
        return int(sys.maxsize)

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                            shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch['noisy'] = batch['noisy'].cuda()   
                batch['clean'] = batch['clean'].cuda()
                batch['affs'] = batch['affs'].cuda()
                batch['wmap'] = batch['wmap'].cuda()
                batch['seg'] = batch['seg'].cuda()
                batch['label_pretrain'] = batch['label_pretrain'].cuda()
               
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch['clean'] = batch['clean'].cuda()
                batch['noisy'] = batch['noisy'].cuda()
                batch['affs'] = batch['affs'].cuda()
                batch['wmap'] = batch['wmap'].cuda()
                batch['seg'] = batch['seg'].cuda()
                batch['label_pretrain'] = batch['label_pretrain'].cuda()
               
            return batch

if __name__ == '__main__':
    import cv2
    from utils.show import draw_fragments_2d
    dst = TrainingDataSet(data_name='CREMIA')
    # dst = Validation(data_name='CREMIA')
    out_path = './data_temp'
    if not osp.exists(out_path):
        os.makedirs(out_path)
    start = time.time()
    for i, data in enumerate(dst):
        if i < 50:
            print(i)
            img, affs, wmap, _ = data
            img = (img.numpy() * 255).astype(np.uint8).squeeze()
            affs = (affs.numpy() * 255).astype(np.uint8).squeeze()[-1]
            wmap = (wmap.numpy() * 255).astype(np.uint8).squeeze()[-1]
            concat = np.concatenate([img, affs, wmap], axis=1)
            Image.fromarray(concat).save(osp.join(out_path, str(i).zfill(4)+'.png'))
        else:
            break
        # print(i)
        # img, lb = data
        # img = (img.numpy() * 255).astype(np.uint8)
        # lb = (lb.numpy() * 255).astype(np.uint64)
        # img = img.squeeze()
        # img = img[:, :, np.newaxis]
        # img = np.repeat(img, 3, 2)
        # lb_color = draw_fragments_2d(lb)
        # concat = np.concatenate([img, lb_color], axis=1)
        # # Image.fromarray(concat).save(osp.join(out_path, str(i).zfill(4)+'.png'))
        # cv2.imwrite(osp.join(out_path, str(i).zfill(4)+'.png'), concat)
    print('COST TIME:', time.time() - start)
    print('Done')