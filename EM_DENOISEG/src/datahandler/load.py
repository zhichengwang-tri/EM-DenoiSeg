
import time
import torch
from .data_provider import Provider,Validation_noisy_concluded

def load_dataset(cfg):
    print('Caching datasets ... ', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_data = Validation_noisy_concluded(root_path=cfg.DATA.root_path,
                                data_name=cfg.DATA.data_name,
                                num_train=cfg.DATA.num_train,
                                num_valid=cfg.DATA.num_valid,
                                num_test=cfg.DATA.num_test,
                                crop_size=list(cfg.DATA.crop_size),
                                padding=0)
        valid_provider = torch.utils.data.DataLoader(valid_data,
                                           batch_size=1,
                                           shuffle=False)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider