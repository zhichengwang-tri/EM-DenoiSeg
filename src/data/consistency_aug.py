
import cv2
import torch
import random
import numpy as np


def tensor2img(img):
    img = img.numpy()
    std = np.asarray([0.229, 0.224, 0.225])
    std = std[:, np.newaxis, np.newaxis]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[:, np.newaxis, np.newaxis]
    img = img * std + mean
    return img


def img2tensor(img):
    std = np.asarray([0.229, 0.224, 0.225])
    std = std[:, np.newaxis, np.newaxis]
    mean = np.asarray([0.485, 0.456, 0.406])
    mean = mean[:, np.newaxis, np.newaxis]
    img = (img.astype(np.float32) - mean) / std
    return torch.from_numpy(img.astype(np.float32))


def simple_augment(data, rule):
    assert np.size(rule) == 3
    assert data.ndim == 2
    # x reflection
    if rule[0]:
        data = data[:, ::-1]
    # y reflection
    if rule[1]:
        data = data[::-1, :]
    # transpose in xy
    if rule[2]:
        data = np.transpose(data, (1, 0))
    return data


def simple_augment_torch(data, rule):
    assert np.size(rule) == 3
    assert len(data.shape) == 3
    # x reflection
    if rule[0]:
        data = torch.flip(data, [2])
    # y reflection
    if rule[1]:
        data = torch.flip(data, [1])
    # transpose in xy
    if rule[2]:
        data = data.permute(0, 2, 1)
    return data


def simple_augment_reverse_torch(data, rule):
    assert np.size(rule) == 3
    assert len(data.shape) == 3
    # transpose in xy
    if rule[2]:
        data = data.permute(0, 2, 1)
    # y reflection
    if rule[1]:
        data = torch.flip(data, [1])
    # x reflection
    if rule[0]:
        data = torch.flip(data, [2])
    return data


def convert_consistency_flip(gt, rules):
    B, C, H, W = gt.shape
    gt = gt.detach().clone()
    rules = rules.data.cpu().numpy().astype(np.uint8)
    out_gt = []
    for k in range(B):
        gt_temp = gt[k]
        rule = rules[k]
        gt_temp = simple_augment_reverse_torch(gt_temp, rule)
        out_gt.append(gt_temp)
    out_gt = torch.stack(out_gt, dim=0)
    return out_gt


def add_gauss_noise(imgs, std=0.01, norm_mode='norm'):
    gaussian = np.random.normal(0, std, (imgs.shape))
    imgs = imgs + gaussian
    if norm_mode == 'norm':
        imgs = (imgs-np.min(imgs)) / (np.max(imgs)-np.min(imgs))
    elif norm_mode == 'trunc':
        imgs[imgs<0] = 0
        imgs[imgs>1] = 1
    else:
        raise NotImplementedError
    return imgs


def add_gauss_blur(imgs, kernel_size=5, sigma=0):
    imgs = cv2.GaussianBlur(imgs, (kernel_size,kernel_size), sigma)
    imgs = np.asarray(imgs, dtype=np.float32)
    imgs[imgs < 0] = 0
    imgs[imgs > 1] = 1
    return imgs


def add_intensity(imgs, contrast_factor=0.1, brightness_factor=0.1):
    imgs *= 1 + (np.random.rand() - 0.5) * contrast_factor
    imgs += (np.random.rand() - 0.5) * brightness_factor
    imgs = np.clip(imgs, 0, 1)
    imgs **= 2.0**(np.random.rand()*2 - 1)
    return imgs


def add_mask(imgs, mask_counts=20, mask_size=10):
    mean = np.mean(imgs)
    mask = np.ones_like(imgs, dtype=np.float32)
    crop_size = list(imgs.shape)
    for k in range(mask_counts):
        my = random.randint(0, crop_size[0]-mask_size)
        mx = random.randint(0, crop_size[1]-mask_size)
        mask[my:my+mask_size, mx:mx+mask_size] = 0
    imgs = imgs * mask + (1-mask) * mean
    return imgs


class Filp(object):
    def __init__(self):
        super(Filp, self).__init__()

    def __call__(self, data):
        rule = np.random.randint(2, size=3)
        # data = simple_augment_torch(data, rule)
        data = simple_augment(data, rule)
        return data, rule


class GaussNoise(object):
    def __init__(self, min_std=0.01, max_std=0.2, norm_mode='trunc'):
        super(GaussNoise, self).__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.norm_mode = norm_mode

    def __call__(self, data):
        std = random.uniform(self.min_std, self.max_std)
        data = add_gauss_noise(data, std=std, norm_mode=self.norm_mode)
        return data


class GaussBlur(object):
    def __init__(self, min_kernel=3, max_kernel=9, min_sigma=0, max_sigma=2):
        super(GaussBlur, self).__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, data):
        kernel_size = random.randint(self.min_kernel // 2, self.max_kernel // 2)
        kernel_size = kernel_size * 2 + 1
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        data = add_gauss_blur(data, kernel_size=kernel_size, sigma=sigma)
        return data


class Intensity(object):
    def __init__(self, CONTRAST_FACTOR=0.1, BRIGHTNESS_FACTOR=0.1):
        super(Intensity, self).__init__()
        self.CONTRAST_FACTOR = CONTRAST_FACTOR
        self.BRIGHTNESS_FACTOR = BRIGHTNESS_FACTOR
    
    def __call__(self, data):
        data = add_intensity(data, self.CONTRAST_FACTOR, self.BRIGHTNESS_FACTOR)
        return data


class Cutout(object):
    def __init__(self, min_mask_counts=20, max_mask_counts=60, min_mask_size=5, max_mask_size=20):
        super(Cutout, self).__init__()
        self.min_mask_counts = min_mask_counts
        self.max_mask_counts = max_mask_counts
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size

    def __call__(self, data):
        mask_counts = random.randint(self.min_mask_counts, self.max_mask_counts)
        mask_size = random.randint(self.min_mask_size, self.max_mask_size)
        data = add_mask(data, mask_counts, mask_size)
        return data