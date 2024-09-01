import cv2
import numpy as np
from skimage import morphology

def gen_skele(lb, dilate=3):
    # padding
    shift=20
    img_h, img_w = lb.shape
    lb_padding = np.zeros((img_h+2*shift, img_w+2*shift), dtype=np.uint64)
    lb_padding[shift:-shift, shift:-shift] = lb

    ids = np.unique(lb_padding)
    skele = np.zeros_like(lb_padding, dtype=np.uint8)

    for id in ids:
        if id == 0: continue
        # select one instance
        mask = np.zeros_like(lb_padding, dtype=np.uint8)
        mask[lb_padding == id] = 255
        # find its contours
        contours, _ = cv2.findContours(mask, 3, 1)
        cnt = contours[0]
        length = len(contours)
        if length > 1:
            for k in range(1, length):
                cnt = np.concatenate([cnt, contours[k]], axis=0)
        # Crop to reduce the size of image
        x,y,w,h = cv2.boundingRect(cnt)
        x = x - shift // 2; y = y - shift // 2
        w = w + shift; h = h + shift
        crop_mask = mask[y:y+h, x:x+w]
        # {0, 1} used to extract skeleton
        crop_mask = crop_mask // 255
        mask_skele = morphology.skeletonize(crop_mask)
        # mask_skele = cv2.ximgproc.thinning(crop_mask * 255)
        # dilate
        mask_skele = morphology.binary_dilation(mask_skele, morphology.square(dilate))
        # mask_skele = morphology.dilation(mask_skele, morphology.square(dilate))
        skele[y:y+h, x:x+w] += mask_skele
    skele = skele[shift:-shift, shift:-shift]
    skele[skele > 1] = 1
    skele = skele.astype(np.int32)
    return skele
