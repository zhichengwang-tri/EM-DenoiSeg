'''
Description: 
Author: weihuang
Date: 2021-11-15 17:37:00
LastEditors: Please set LastEditors
LastEditTime: 2022-05-30 16:55:52
'''
import os
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# from utils.pre_processing import division_array, image_concatenate

def draw_fragments_2d(pred, print_num=True):
    m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    if print_num: print("the neurons number of pred is %d" % size)
    color_pred = np.zeros([m, n, 3])
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,i] = color_val[idx]
    color_pred = color_pred
    return color_pred

def draw_fragments_3d(pred):
    d,m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the neurons number of pred is %d" % size)
    color_pred = np.zeros([d, m, n, 3])
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,:,i] = color_val[idx]
    color_pred = color_pred
    return color_pred

def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

def convert2png(img):
    img = img.data.cpu().numpy()
    img = np.squeeze(img)
    img[img<0] = 0
    img[img>1] = 1
    img = (img * 255).astype(np.uint8)
    return img

def show_training(iters, imgs, pred, save_path, out_channel=2):
    converted_images = []
    for img in imgs:
        img = convert2png(img)
        converted_images.append(img)
    if out_channel > 1:
        pred = torch.argmax(pred, dim=0).float()
    pred = convert2png(pred)
    concat = np.concatenate(converted_images + [pred], axis=1)
    Image.fromarray(concat).save(os.path.join(save_path, str(iters).zfill(6)+'.png'))


def show_valid(iters, pred, label, save_path):
    label = (label * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    concat = np.concatenate([pred, label], axis=1)
    Image.fromarray(concat).save(os.path.join(save_path, str(iters).zfill(6)+'.png'))


def show_test(iters, img, label, pred, label_ins, pred_ins, save_path):
    img = convert2png(img)
    label = (label * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    img = np.repeat(img[:,:,np.newaxis], 3, 2)
    label = np.repeat(label[:,:,np.newaxis], 3, 2)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    label_color = draw_fragments_2d(label_ins)
    pred_color = draw_fragments_2d(pred_ins)
    concat1 = np.concatenate([img, pred, label], axis=1)
    concat2 = np.concatenate([img, pred_color, label_color], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    cv2.imwrite(os.path.join(save_path, str(iters).zfill(6)+'.png'), concat)

def show_test2(iters, img, pred, pred_ins, label_ins, save_path):
    img = convert2png(img)
    pred = (pred * 255).astype(np.uint8)
    img = np.repeat(img[:,:,np.newaxis], 3, 2)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    pred_color = draw_fragments_2d(pred_ins)
    label_color = draw_fragments_2d(label_ins)
    concat1 = np.concatenate([img, pred], axis=1)
    concat2 = np.concatenate([pred_color, label_color], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    cv2.imwrite(os.path.join(save_path, str(iters).zfill(6)+'.png'), concat)

def show_test3(iters, img, pred, skele, emb, pred_ins, label_ins, save_path):
    img = convert2png(img)
    pred = (pred * 255).astype(np.uint8)
    skele = (skele * 255).astype(np.uint8)
    img = np.repeat(img[:,:,np.newaxis], 3, 2)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    skele = np.repeat(skele[:,:,np.newaxis], 3, 2)
    emb_color = embedding_pca(emb)
    pred_color = draw_fragments_2d(pred_ins)
    label_color = draw_fragments_2d(label_ins)
    concat1 = np.concatenate([img, pred, skele], axis=1)
    concat2 = np.concatenate([emb_color, pred_color, label_color], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    cv2.imwrite(os.path.join(save_path, str(iters).zfill(6)+'.png'), concat)

def show_test4(iters, img, pred, emb, pred_ins, save_path):
    img = convert2png(img)
    pred = (pred * 255).astype(np.uint8)
    img = np.repeat(img[:,:,np.newaxis], 3, 2)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    emb_color = embedding_pca(emb)
    pred_color = draw_fragments_2d(pred_ins)
    concat1 = np.concatenate([img, pred], axis=1)
    concat2 = np.concatenate([emb_color, pred_color], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    cv2.imwrite(os.path.join(save_path, str(iters).zfill(6)+'.png'), concat)

def show_test_skele(iters, img, label, pred, skele, label_ins, pred_ins, save_path):
    img = convert2png(img)
    label = (label * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    skele = (skele * 255).astype(np.uint8)
    img = np.repeat(img[:,:,np.newaxis], 3, 2)
    label = np.repeat(label[:,:,np.newaxis], 3, 2)
    pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    skele = np.repeat(skele[:,:,np.newaxis], 3, 2)
    label_color = draw_fragments_2d(label_ins)
    pred_color = draw_fragments_2d(pred_ins)
    concat1 = np.concatenate([img, pred, label], axis=1)
    concat2 = np.concatenate([skele, pred_color, label_color], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    # Image.fromarray(concat).save(os.path.join(save_path, str(iters).zfill(6)+'.png'))
    cv2.imwrite(os.path.join(save_path, str(iters).zfill(6)+'.png'), concat)
def show_training_allresults(iters,
                            cimg,
                            clabel,
                            cpred,
                            aimg,
                            alabel,
                            apred,
                            dlabel,
                            dpred,
                            ccross,
                            across,
                            save_path,
                            tag='s'):
    cimg = convert2png(cimg)
    aimg = convert2png(aimg)
    clabel = convert2png(clabel)
    alabel = convert2png(alabel)
    dlabel = convert2png(dlabel)
    ccross = convert2png(ccross)
    across = convert2png(across)
    cpred = torch.argmax(cpred, dim=0).float()
    cpred = convert2png(cpred)
    apred = torch.argmax(apred, dim=0).float()
    apred = convert2png(apred)
    dpred = torch.argmax(dpred, dim=0).float()
    dpred = convert2png(dpred)
    concat1 = np.concatenate([cimg, clabel, cpred, dpred, ccross], axis=1)
    concat2 = np.concatenate([aimg, alabel, apred, dlabel, across], axis=1)
    concat = np.concatenate([concat1, concat2], axis=0)
    Image.fromarray(concat).save(os.path.join(save_path, str(iters).zfill(6)+'_%s.png' % tag))

# def show_test(preds, labels, raw_path, save_path):
#     num = labels.shape[0]
#     for k in range(num):
#         img = np.asarray(Image.open(os.path.join(raw_path, str(k).zfill(3)+'.png')))
#         pred = preds[k]
#         label = labels[k]
#         img = draw_label(img, pred, label)
#         cv2.imwrite(os.path.join(save_path, str(k).zfill(3)+'.png'), img)

def draw_label(img, pred, label):
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    if pred.max() <= 1:
        pred = (pred * 255).astype(np.uint8)
    else:
        pred = pred.astype(np.uint8)
    if label.max() <= 1:
        label = (label * 255).astype(np.uint8)
    else:
        label = label.astype(np.uint8)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        img = np.repeat(img, 3, 2)
    contours_lb, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours_lb, -1, (0,0,255), 2)
    contours_pred, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours_pred, -1, (0,255,0), 2)
    return img

def embedding_pca(embeddings, n_components=3, as_rgb=True):
    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')
    
    embed_flat = np.transpose(embed_flat, (1, 2, 0))

    return embed_flat

def val_show_dn(iters, pred_dn, dn_truth, noise, valid_path,k):
    os.makedirs(os.path.join(valid_path, 'dn'), exist_ok=True)
    pred_dn = pred_dn[:,:,np.newaxis]
    pred_dn = np.repeat(pred_dn, 3, 2)
    dn_truth = dn_truth[:,:,np.newaxis]
    dn_truth = np.repeat(dn_truth, 3, 2)
    noise = noise[:,:,np.newaxis]
    noise = np.repeat(noise, 3, 2)
    pred_dn = ((pred_dn + 1 )/2 * 255).astype(np.uint8)
    dn_truth = ((dn_truth +1)/2 * 255).astype(np.uint8)
    noise = ((noise+1)/2 * 255).astype(np.uint8)
    im_cat3 = np.concatenate([pred_dn, noise, dn_truth], axis=1)
    Image.fromarray(im_cat3).save(os.path.join(valid_path, 'dn','{0}_{1}.png'.format(iters,k)))

def show_valid_seg(iters, inputs, pred, target, valid_path, k):
    os.makedirs(os.path.join(valid_path, 'seg'), exist_ok=True)
    inputs = inputs[:,:,np.newaxis]
    inputs = np.repeat(inputs,3,2)
    inputs = ((inputs+1)/2*255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8) 
    # pred = np.repeat(pred[:,:,np.newaxis], 3, 2)
    pred_color = draw_fragments_2d(pred, print_num=False)
    label_color = draw_fragments_2d(target, print_num=False)
    im_cat = np.concatenate([inputs, pred_color, label_color], axis=1)
    im_cat = im_cat.squeeze()
    Image.fromarray(np.array(np.uint8(im_cat))).save(os.path.join(valid_path, 'seg','{0}_{1}.png'.format(iters,k)))

def show_valid_seg_v2(iters,  pred, target, valid_path, k):
    os.makedirs(os.path.join(valid_path, 'seg'), exist_ok=True)
    pred = (pred * 255).astype(np.uint8) 
    pred_color = draw_fragments_2d(pred, print_num=False)
    label_color = draw_fragments_2d(target, print_num=False)
    im_cat = np.concatenate([pred_color, label_color], axis=1)
    im_cat = im_cat.squeeze()
    Image.fromarray(np.array(np.uint8(im_cat))).save(os.path.join(valid_path, 'seg','{0}_{1}.png'.format(iters,k)))
