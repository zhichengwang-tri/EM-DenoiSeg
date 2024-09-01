from tkinter.messagebox import NO
import cv2
import mahotas
import numpy as np
from numba import njit, jit
import scipy.sparse as ss
from scipy import ndimage
from skimage import morphology
from skimage import measure
from skimage.segmentation import find_boundaries
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes

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

# 清除奇异label--input 2D label
def remove_noise_label(label, num=100):
    ids, count = np.unique(label, return_counts=True)
    for i, id in enumerate(ids):
        if id == 0:
            continue
        if count[i] < num:
            label[label == id] = 0
    return label

# merge small id
def merge_small_id(label, min_size=20):
    ids, count = np.unique(label, return_counts=True)
    for i, cnt in enumerate(count):
        id = ids[i]
        if id == 0: continue
        direction = True # True is left direction
        if cnt > min_size: continue
        x_list, y_list = np.where(label == id)
        x = x_list[0]
        y = y_list[0]
        if x < min_size: direction = False
        while True:
            if direction: x -= 1
            else: x += 1
            new_id = label[x, y]
            if new_id != id:
                label[label == id] = new_id
                break
    return label

def merge_small_id2(label, min_size=20):
    slices = find_objects(label)
    for i, slc in enumerate(slices):
        if slc is None: continue
        msk = label[slc] == (i+1)
        npix = msk.sum()
        if npix > min_size: continue
        sr, sc = slc
        sr_start, sr_stop = sr.start, sr.stop
        sc_start, sc_stop = sc.start, sc.stop
        x_list, y_list = np.where(msk)
        x = x_list[0] + sr_start
        y = y_list[0] + sc_start
        direction = True
        if x < min_size: direction = False
        while True:
            if direction: x -= 1
            else: x += 1
            new_id = label[x, y]
            if new_id != (i+1):
                label[slc][msk] = new_id
                break
    return label

def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled

def im2col(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N + np.arange(0,N-BSZ[1]+1,stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4): 
    # "we preprocessed the ground truth seg such that any voxel centered on a 3 × 3 × 1 window containing 
    # more than one positive segment ID (zero is reserved for background) is marked as background."
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape
    if len(sz)==3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z],((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
            p0=patch.max(axis=1)
            patch[patch==0] = mm+1
            p1=patch.min(axis=1)
            seg[z] =seg[z]*((p0==p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg,((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis = 1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg

def genSegMalis(gg3,iter_num): # given input seg map, widen the seg border
    gg3_dz = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dz[1:,:,:] = (np.diff(gg3,axis=0))
    gg3_dy = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dy[:,1:,:] = (np.diff(gg3,axis=1))
    gg3_dx = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dx[:,:,1:] = (np.diff(gg3,axis=2))
    gg3g = ((gg3_dx+gg3_dy)>0)
    #stel=np.array([[1, 1],[1,1]]).astype(bool)
    stel=np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(bool)
    #stel=np.array([[1,1,1,1],[1, 1, 1, 1],[1,1,1,1],[1,1,1,1]]).astype(bool)
    gg3gd=np.zeros(gg3g.shape)
    for i in range(gg3g.shape[0]):
        gg3gd[i,:,:]=binary_dilation(gg3g[i,:,:],structure=stel,iterations=iter_num)
    out = gg3.copy()
    out[gg3gd==1]=0
    return out

def weight_binary_ratio(label, mask=None, alpha=1.0):
    """Binary-class rebalancing."""
    # input: numpy tensor
    # weight for smaller class is 1, the bigger one is at most 20*alpha
    if label.max() == label.min(): # uniform weights for single-label volume
        weight_factor = 1.0
        weight = np.ones_like(label, np.float32)
    else:
        label = (label!=0).astype(int)
        if mask is None:
            weight_factor = float(label.sum()) / np.prod(label.shape)
        else:
            weight_factor = float((label*mask).sum()) / mask.sum()
        weight_factor = np.clip(weight_factor, a_min=5e-2, a_max=0.99)

        if weight_factor > 0.5:
            weight = label + alpha*weight_factor/(1-weight_factor)*(1-label)
        else:
            weight = alpha*(1-weight_factor)/weight_factor*label + (1-label)

        if mask is not None:
            weight = weight*mask

    return weight.astype(np.float32)

def find_em_boundary(label):
    boundary = np.ones_like(label, dtype=np.float32)
    ids = np.unique(label)
    for id in ids:
        if id == 0:
            boundary[label == 0] = 0
        else:
            tmp = np.zeros_like(label)
            tmp[label == id] = 1
            if np.sum(tmp) > 100:
                tmp_bound = find_boundaries(tmp != 0, mode='outer')
                boundary[tmp_bound == 1] = 0
    return boundary

# 根据连通域过分割label
def overseg_label(label):
    over_label = np.zeros_like(label)
    ids = np.unique(label)
    ite = 1
    for id in ids:
        if id == 0:
            continue
        tmp = np.zeros_like(label)
        tmp[label == id] = 1
        img, num = measure.label(tmp, connectivity=2, background=0, return_num=True)
        for k in range(num):
            over_label[img == (k + 1)] = ite
            ite += 1
    return over_label

def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    (might have issues at borders between cells, todo: check and fix)
    
    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
        
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

def gen_skele_boundary(lb, dilate=3):
    # padding
    shift=20
    img_h, img_w = lb.shape
    lb_padding = np.zeros((img_h+2*shift, img_w+2*shift), dtype=np.uint64)
    lb_padding[shift:-shift, shift:-shift] = lb

    ids, counts = np.unique(lb_padding, return_counts=True)
    skele = np.zeros_like(lb_padding, dtype=np.uint8)
    boundary = np.zeros_like(lb_padding, dtype=np.uint8)

    for id in ids:
        if id == 0:
            boundary[lb_padding == 0] = 1
        else:
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
            mask_bound = find_boundaries(crop_mask != 0, mode='outer')
            mask_skele = morphology.skeletonize(crop_mask)
            # mask_skele = cv2.ximgproc.thinning(crop_mask * 255)
            # dilate
            mask_skele = morphology.binary_dilation(mask_skele, morphology.square(dilate))
            # mask_skele = morphology.dilation(mask_skele, morphology.square(dilate))
            skele[y:y+h, x:x+w] += mask_skele
            boundary[y:y+h, x:x+w] += mask_bound
    skele = skele[shift:-shift, shift:-shift]
    boundary = boundary[shift:-shift, shift:-shift]
    skele[skele > 1] = 1
    boundary[boundary > 1] = 1
    skele = skele.astype(np.float32)
    boundary = boundary.astype(np.float32)
    return skele, boundary

def boundaryMap(lb, padding=10):
    # padding
    if padding > 0: lb = np.pad(lb, ((padding, padding)), mode='constant')

    ids, counts = np.unique(lb, return_counts=True)
    boundary = np.zeros_like(lb, dtype=np.float32)

    for id in ids:
        if id == 0: boundary[lb == 0] = 1
        else:
            # select one instance
            mask = np.zeros_like(lb, dtype=np.uint8)
            mask[lb == id] = 255
            # find its contours
            contours, _ = cv2.findContours(mask, 3, 1)
            cnt = contours[0]
            length = len(contours)
            if length > 1:
                for k in range(1, length):
                    cnt = np.concatenate([cnt, contours[k]], axis=0)
            # Crop to reduce the size of image
            x,y,w,h = cv2.boundingRect(cnt)
            x = x - padding // 2; y = y - padding // 2
            w = w + padding; h = h + padding
            crop_mask = mask[y:y+h, x:x+w]
            # {0, 1} used to extract skeleton
            crop_mask = crop_mask // 255
            mask_bound = find_boundaries(crop_mask != 0, mode='outer')
            boundary[y:y+h, x:x+w] += mask_bound
    if padding > 0: boundary = boundary[padding:-padding, padding:-padding]
    boundary[boundary > 1] = 1
    boundary = boundary.astype(np.float32)
    return boundary

def distanceMap(lb, ignoreSize=100, norm=True, padding=10):
    if padding > 0: lb = np.pad(lb, ((padding, padding)), mode='constant')
    ids, counts = np.unique(lb, return_counts=True)
    disMap = np.zeros_like(lb, dtype=np.float32)

    for i, id in enumerate(ids):
        if id == 0: continue
        if counts[i] < ignoreSize: continue
        # select one instance
        mask = np.zeros_like(lb, dtype=np.uint8)
        mask[lb == id] = 255
        # find its contours
        contours, _ = cv2.findContours(mask, 3, 1)
        cnt = contours[0]
        length = len(contours)
        if length > 1:
            for k in range(1, length):
                cnt = np.concatenate([cnt, contours[k]], axis=0)
        # Crop to reduce the size of image
        x,y,w,h = cv2.boundingRect(cnt)
        x = x - padding // 2; y = y - padding // 2
        w = w + padding; h = h + padding
        crop_mask = mask[y:y+h, x:x+w]
        # {0, 1} used to extract distance map
        crop_mask = (crop_mask // 255).astype(np.bool)
        mask_dt = ndimage.distance_transform_edt(crop_mask)
        if norm: mask_dt = (mask_dt - mask_dt.min()) / (mask_dt.max() - mask_dt.min())
        disMap[y:y+h, x:x+w] += mask_dt
    if padding > 0: disMap = disMap[padding:-padding, padding:-padding]
    disMap = disMap.astype(np.float32)
    return disMap

def skeletonMap(lb, dilate=3):
    ids = np.unique(lb)
    skeleMap = np.zeros_like(lb, dtype=np.float32)

    for i, id in enumerate(ids):
        if id == 0: continue
        # select one instance
        mask = np.zeros_like(lb, dtype=np.uint8)
        mask[lb == id] = 255
        # find its contours
        contours, _ = cv2.findContours(mask, 3, 1)
        cnt = contours[0]
        length = len(contours)
        if length > 1:
            for k in range(1, length):
                cnt = np.concatenate([cnt, contours[k]], axis=0)
        # Crop to reduce the size of image
        x,y,w,h = cv2.boundingRect(cnt)
        crop_mask = mask[y:y+h, x:x+w]
        # {0, 1} used to extract distance map
        crop_mask = crop_mask // 255
        mask_skele = morphology.skeletonize(crop_mask)
        mask_skele = morphology.binary_dilation(mask_skele, morphology.square(dilate))
        skeleMap[y:y+h, x:x+w] += mask_skele
    skeleMap = skeleMap.astype(np.float32)
    return skeleMap

def array_distance(arr1, arr2):
    '''
    计算两个数组里, 每任意两个点之间的L2距离
    arr1 和 arr2 都必须是numpy数组
    且维度分别为 m x 2, n x 2
    输出数组的维度为 m x n
    '''
    m, _ = arr1.shape
    n, _ = arr2.shape
    arr1_power = np.power(arr1, 2)
    arr1_power_sum = arr1_power[:, 0] + arr1_power[:, 1]
    arr1_power_sum = np.tile(arr1_power_sum, (n, 1))
    arr1_power_sum = arr1_power_sum.T
    arr2_power = np.power(arr2, 2)
    arr2_power_sum = arr2_power[:, 0] + arr2_power[:, 1]
    arr2_power_sum = np.tile(arr2_power_sum, (m, 1))
    dis = arr1_power_sum + arr2_power_sum - (2 * np.dot(arr1, arr2.T))
    dis = np.sqrt(dis)
    return dis

# def multiSeedMap(lb, ignoreSize=100, norm=True, padding=10, regSize=51):
#     if padding > 0: lb = np.pad(lb, ((padding, padding)), mode='constant')
#     ids, counts = np.unique(lb, return_counts=True)
#     seedMap = np.zeros_like(lb, dtype=np.float32)
#     Bc = np.ones((regSize, regSize))

#     for i, id in enumerate(ids):
#         if id == 0 or counts[i] < ignoreSize:
#             seedMap[lb==id] = 1.0
#             continue
#         # select one instance
#         mask = np.zeros_like(lb, dtype=np.uint8)
#         mask[lb == id] = 255
#         # find its contours
#         contours, _ = cv2.findContours(mask, 3, 1)
#         cnt = contours[0]
#         length = len(contours)
#         if length > 1:
#             for k in range(1, length):
#                 cnt = np.concatenate([cnt, contours[k]], axis=0)
#         # Crop to reduce the size of image
#         x, y, w, h = cv2.boundingRect(cnt)
#         x = x - padding // 2; y = y - padding // 2
#         w = w + padding; h = h + padding
#         crop_mask = mask[y:y+h, x:x+w]
#         # {0, 1} used to extract distance map
#         crop_mask = crop_mask // 255
#         mask_dt = ndimage.distance_transform_edt(crop_mask)
#         if norm: mask_dt = (mask_dt - mask_dt.min()) / (mask_dt.max() - mask_dt.min())
#         maxima = mahotas.regmax(mask_dt, Bc=Bc)
#         arr_mask = np.asarray(np.where(crop_mask != 0)).T
#         arr_seeds = np.asarray(np.where(maxima != 0)).T
#         seedDis = array_distance(arr_seeds, arr_mask)
#         seedDis = np.min(seedDis, axis=0)
#         seedDis = (seedDis - seedDis.min()) / (seedDis.max() - seedDis.min())
#         x_list = arr_mask[:,0]
#         y_list = arr_mask[:,1]
#         mask_dis = ss.coo_matrix((seedDis, (x_list, y_list)),shape=(h, w))
#         seedMap[y:y+h, x:x+w] += mask_dis
#     if padding > 0: seedMap = seedMap[padding:-padding, padding:-padding]
#     seedMap = seedMap.astype(np.float32)
#     return seedMap

def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5)/2
    return md, counts**0.5

def multiSeedMap(lb, ignoreSize=100, norm=True, padding=2, regSize=9):
    if padding > 0: lb = np.pad(lb, ((padding, padding)), mode='constant')
    seedMap = np.zeros_like(lb, dtype=np.float32)
    Bc = np.ones((regSize, regSize))
    slices = ndimage.find_objects(lb)
    seedMap[lb==0] = 1

    for i, si in enumerate(slices):
        if si is None: continue
        sr, sc = si
        sr_start, sr_stop = sr.start - padding, sr.stop + padding
        sc_start, sc_stop = sc.start - padding, sc.stop + padding
        # subMask = np.zeros_like(lb[sr, sc], dtype=np.uint8)
        # subMask[lb[sr, sc] == (i+1)] = 1
        subMask = np.zeros_like(lb[sr_start:sr_stop, sc_start:sc_stop], dtype=np.uint8)
        subMask[lb[sr_start:sr_stop, sc_start:sc_stop] == (i+1)] = 1
        h, w = subMask.shape
        if np.sum(subMask) < ignoreSize: continue
        mask_dt = ndimage.distance_transform_edt(subMask)
        if norm: mask_dt = (mask_dt - mask_dt.min()) / (mask_dt.max() - mask_dt.min())
        maxima = mahotas.regmax(mask_dt, Bc=Bc)
        arr_mask = np.asarray(np.where(subMask != 0)).T
        arr_seeds = np.asarray(np.where(maxima != 0)).T
        seedDis = array_distance(arr_seeds, arr_mask)
        seedDis = np.min(seedDis, axis=0)
        seedDis = (seedDis - seedDis.min()) / (seedDis.max() - seedDis.min())
        x_list = arr_mask[:,0]
        y_list = arr_mask[:,1]
        mask_dis = ss.coo_matrix((seedDis, (x_list, y_list)), shape=(h, w))
        # seedMap[sr, sc] += mask_dis
        seedMap[sr_start:sr_stop, sc_start:sc_stop] += mask_dis
    if padding > 0: seedMap = seedMap[padding:-padding, padding:-padding]
    return seedMap

# def multiSeedMap_2(lb, ignoreSize=100, norm=True, padding=10, regSize=51):
#     seedMap = np.zeros_like(lb, dtype=np.float32)
#     Bc = np.ones((regSize, regSize))
#     slices = ndimage.find_objects(lb)

#     for i, si in enumerate(slices):
#         if si is None: continue
#         sr, sc = si
#         subMask = np.zeros_like(lb[sr, sc], dtype=np.uint8)
#         subMask[lb[sr, sc] == (i+1)] = 1
#         h, w = subMask.shape
#         if np.sum(subMask) < ignoreSize: continue
#         mask_dt = ndimage.distance_transform_edt(subMask)
#         if norm: mask_dt = (mask_dt - mask_dt.min()) / (mask_dt.max() - mask_dt.min())
#         maxima = mahotas.regmax(mask_dt, Bc=Bc)
#         arr_mask = np.asarray(np.where(subMask != 0)).T
#         arr_seeds = np.asarray(np.where(maxima != 0)).T
#         seedDis = array_distance(arr_seeds, arr_mask)
#         seedDis = np.min(seedDis, axis=0)
#         seedDis = (seedDis - seedDis.min()) / (seedDis.max() - seedDis.min())
#         x_list = arr_mask[:,0]
#         y_list = arr_mask[:,1]
#         mask_dis = ss.coo_matrix((seedDis, (x_list, y_list)), shape=(h, w))
#         seedMap[sr, sc] += mask_dis
#     return seedMap


# def multiSeedMap(lb, ignoreSize=100, norm=True, padding=10, regSize=51):
#     if padding > 0: lb = np.pad(lb, ((padding, padding)), mode='constant')
#     ids, counts = np.unique(lb, return_counts=True)
#     seedMap = np.zeros_like(lb, dtype=np.float32)
#     Bc = np.ones((regSize, regSize))
#     h, w = lb.shape

#     for i, id in enumerate(ids):
#         if id == 0: 
#             seedMap[lb==0] = 1.0
#             continue
#         if counts[i] < ignoreSize: continue
#         # select one instance
#         mask = np.zeros_like(lb, dtype=np.uint8)
#         mask[lb == id] = 1
#         mask_dt = ndimage.distance_transform_edt(mask)
#         if norm: mask_dt = (mask_dt - mask_dt.min()) / (mask_dt.max() - mask_dt.min())
#         maxima = mahotas.regmax(mask_dt, Bc=Bc)
#         arr_mask = np.asarray(np.where(mask != 0)).T
#         arr_seeds = np.asarray(np.where(maxima != 0)).T
#         seedDis = array_distance(arr_seeds, arr_mask)
#         seedDis = np.min(seedDis, axis=0)
#         seedDis = (seedDis - seedDis.min()) / (seedDis.max() - seedDis.min())
#         seedMap += ss.coo_matrix((seedDis, (arr_mask[:,0], arr_mask[:,1])), shape=(h, w))
#     if padding > 0: seedMap = seedMap[padding:-padding, padding:-padding]
#     seedMap = seedMap.astype(np.float32)
#     return seedMap