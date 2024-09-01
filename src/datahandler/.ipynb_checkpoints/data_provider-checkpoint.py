'''
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
from src.utils.pre_processing import normalization2, approximate_image, cropping
from src.data.data_aug import aug_img_lab
from src.utils.affinity_ours import multi_offset, gen_affs_ours
from src.data.data_segmentation import weight_binary_ratio
from src.data.pre_processing import crop

class TrainingDataSet(data.Dataset):
    def __init__(self, root_path='/braindat/lab/chenyd/DATASET',
                    data_name='CREMI',
                    num_train=1,
                    crop_size=[256, 256],
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
        raw_path = osp.join('/braindat/lab/wangzc/dataset/cremiC/cremic_files/cremiC_inputs_interp.h5')
        noisy_path = osp.join('/braindat/lab/wangzc/dataset/cremiC/cremic_files/cremiC_inputs_interp_noisy_50_20.h5')
        label_path = osp.join('/braindat/lab/wangzc/dataset/cremiC/cremic_files/boundary_cremiC_gt.npy')
        print('Load raw:', raw_path)
        f = h5py.File(raw_path, 'r')
        self.total_raw = f['main'][:]
        f.close()
        print("Load noisy", noisy_path)
        f = h5py.File(noisy_path, 'r')
        self.total_noisy = f['main'][:]
        f.close()
        self.total_label = np.load(label_path)
        
        self.training_raw = self.total_raw[:self.num_train]
        self.training_noisy = self.total_noisy[:self.num_train]
        self.training_label = self.total_label[:self.num_train]
        print('Training shape:', self.training_raw.shape)
        self.num_training = self.training_raw.shape[0]

    def __len__(self):
        return int(sys.maxsize)
        # return 10

    def __getitem__(self, idx):
        k = random.randint(0, self.num_training - 1)
        img = self.training_raw[k]
        noisy = self.training_noisy[k]
        label = self.training_label[k]
        size = img.shape
        y_loc = random.randint(0, size[0] - self.crop_size[0] - 1)
        x_loc = random.randint(0, size[1] - self.crop_size[1] - 1)
        img = img[y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        noisy = noisy[y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        label = label[y_loc:y_loc+self.crop_size[0], x_loc:x_loc+self.crop_size[1]]
        img = img.astype(np.float32) / 255.0
        noisy = noisy.astype(np.float32) / 255.0
        label = label.astype(np.uint64)
        img = np.expand_dims(img, axis=0)  # add additional dimension
        noisy = np.expand_dims(noisy , axis=0)
        label =np.expand_dims(label, axis = 0)
        img = torch.from_numpy(img).float()
        noisy = torch.from_numpy(noisy).float()
        label = torch.from_numpy(label.astype(np.int64)).long()
        
        return {'clean': img,
                'noisy': noisy,
                'boundary': label}
        
       
class Validation(data.Dataset):
    def __init__(self, root_path='../data',
                    data_name='CREMI',
                    mode='valid',
                    num_train=1,
                    num_valid=10,
                    num_test=50,
                    crop_size=[256, 256],
                    padding=0):
        self.padding = padding
        self.root_path = root_path  # ../data
        self.data_name = data_name  # ISBI2012, ISBI2013, CREMIA
        self.crop_size = crop_size  # [256, 256]

        raw_path = osp.join('/braindat/lab/wangzc/dataset/cremiC/cremic_files/cremiC_inputs_interp.h5')
        noisy_path = osp.join('/braindat/lab/wangzc/dataset/cremiC/cremic_files/cremiC_inputs_interp_noisy_50_20.h5')
        label_path = osp.join('/braindat/lab/wangzc/dataset/cremiC/cremic_files/boundary_cremiC_gt.npy')
        print('Load raw:', raw_path)
        f = h5py.File(raw_path, 'r')
        self.total_raw = f['main'][:]
        f.close()
        print("Load noisy", noisy_path)
        f = h5py.File(noisy_path, 'r')
        self.total_noisy = f['main'][:]
        f.close()
        self.total_label = np.load(label_path)
        
        self.valid_raw = self.total_raw[num_train:num_train+num_valid]
        self.valid_noisy = self.total_noisy[num_train:num_train+num_valid]
        self.valid_label = self.total_label[num_train:num_train+num_valid]


    def __len__(self):
        return self.valid_raw.shape[0]

    def __getitem__(self, k):
        
        img = self.valid_raw[k]
        noisy = self.valid_noisy[k]
        lb = self.valid_label[k]
        if self.padding > 0:
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
            noisy = np.pad(noisy, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
            lb = np.pad(lb, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
        img = img.astype(np.float32) / 255.0
        noisy = noisy.astype(np.float32) / 255.0
        lb = lb.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # add additional dimension
        noisy = np.expand_dims(noisy, axis=0)
        img = torch.from_numpy(img).float()
        noisy = torch.from_numpy(noisy).float()
        lb = torch.from_numpy(lb).long()
        return img, lb
    
'''
在validation 阶段，送入noisy数据，首先由denoise网络去噪再经分割网络分割
'''
class Validation_noisy_concluded(data.Dataset):
    def __init__(self, root_path='/braindat/lab/chenyd/DATASET',
                    data_name='CREMI',
                    mode='valid',
                    num_train=75,
                    num_valid=50,
                    num_test =0,
                    crop_size=[256, 256],
                    padding=0,
                ):
        self.padding = padding
        self.root_path = root_path  # ../data
        self.data_name = data_name  # ISBI2012, ISBI2013, CREMIA
        self.crop_size = crop_size  # [256, 256]
        
        # Load data
        raw_path = osp.join(self.root_path, self.data_name, 'cremiC_inputs_interp.h5')
        noisy_path =osp.join("/braindat/lab/wangzc/code/jdas","cremiC_inputs_interp_noisy_25_10.h5")
        label_path = osp.join(self.root_path, self.data_name, 'cremiC_labels.h5')
        print('Load raw:', raw_path)
        f = h5py.File(raw_path, 'r')
        total_raw = f['main'][:]
        f.close()
        print("Load noisy",noisy_path)
        f = h5py.File(noisy_path, 'r')
        total_noisy = f['main'][:]
        f.close()
        print('Load labels:', label_path)
        f = h5py.File(label_path, 'r')
        total_label = f['main'][:]
        f.close()
        if mode == 'valid':
            self.valid_raw = total_raw[num_train:num_train+num_valid]
            self.valid_label = total_label[num_train:num_train+num_valid]
            self.valid_noisy = total_noisy[num_train:num_train+num_valid]
        else:
            self.valid_raw = total_raw[-num_test:]
            self.valid_noisy = total_noisy[-num_test:]
            self.valid_label = total_label[-num_test:]
        print('Valid shape:', self.valid_raw.shape)
    
    def __len__(self):
        return self.valid_raw.shape[0]

    def __getitem__(self, k):
        img = self.valid_raw[k]
        noisy = self.valid_noisy[k]
        lb = self.valid_label[k]
        if self.padding > 0:
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
            noisy = np.pad(noisy, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')    
            lb = np.pad(lb, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')            
        img = normalization2(img.astype(np.float32), max=1, min=0)
        noisy =normalization2(noisy.astype(np.float32),max=1,min=0)
        lb = lb.astype(np.float32)
        img = np.expand_dims(img, axis=0)  # add additional dimension
        img = torch.from_numpy(img).float()
        noisy = np.expand_dims(noisy, axis=0)  # add additional dimension
        noisy = torch.from_numpy(noisy).float()
        lb = torch.from_numpy(lb).long()
        return noisy, img, lb

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
                                        noise_type = cfg.DATA.noise_type,
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
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
                batch[3] = batch[3].cuda()
                batch[4] = batch[4].cuda()
                #add one item 
                batch[5] = batch[5].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
                batch[3] = batch[3].cuda()
                batch[4] = batch[4].cuda()
                #add one item 
                batch[5] = batch[5].cuda()
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