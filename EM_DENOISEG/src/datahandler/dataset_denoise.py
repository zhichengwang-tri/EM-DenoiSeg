import os
import os.path as osp
import numpy as np
import random
import h5py
import sys
import torch
import cv2
import glob
import torch.utils.data as udata
from src.utils.utils_denoise import data_augmentation
from torch.utils import data
from src.data.pre_processing import normalization2, approximate_image, cropping
def normalize(data):
    return data/255.

def crop(img, label=None, patch_size=[1,256,256]):

    def _crop(input_arry):
        x, y, z = input_arry.shape  
        assert (x % patch_size[0] == 0 and y % patch_size[1] == 0), "patch_size[0]和patch[1]不能被x,y整除"
        maxiter = int(np.ceil(z / patch_size[-1]))  

        crops_img = []
        w, h, d = patch_size 
        for i in range(x // w):
            for j in range(y // h):
                for k in range(maxiter - 1):
                    imgt = input_arry[i * w:(i + 1) * w, j * h:(j + 1) * h, k * d:(k + 1) * d]
                    if type(label) != type(None):
                        imgtlabel = label[i * w:(i + 1) * w, j * h:(j + 1) * h, k * d:(k + 1) * d]
                        prob = np.sum(imgtlabel) / (imgtlabel.shape[0] * imgtlabel.shape[1] * imgtlabel.shape[2])
                        if prob < 0.05: 
                            continue
                        crops_img.append(imgt)
                    else:
                        crops_img.append(imgt)
        for i in range(x // w):
            for j in range(y // h):
                imgt = input_arry[i * w:(i + 1) * w, j * h:(j + 1) * h, -d:]
                if type(label) != type(None):
                    imgtlabel = label[i * w:(i + 1) * w, j * h:(j + 1) * h, -d:]
                    prob = np.sum(imgtlabel) / (imgtlabel.shape[0] * imgtlabel.shape[1] * imgtlabel.shape[2])
                    if prob < 0.05: 
                        continue
                    crops_img.append(imgt)
                else:
                    crops_img.append(imgt)

        crops_img = np.array(crops_img)
        return crops_img
    if type(label) == type(None):
        ans = _crop(img)
        print(f'crop_img.shape={ans.shape}')
        return ans
    else:
        ans1, ans2 = _crop(img), _crop(label)
        print(f'crop_img.shape={ans1.shape}, _crop(label).shape={ans2.shape}')
        return ans1, ans2

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)
