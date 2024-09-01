#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python dn_1stage.py -c=dn_1stage_c_ridnet&
CUDA_VISIBLE_DEVICES=2,3 python dn_1stage.py -c=dn_1stage_c_dncnn&
CUDA_VISIBLE_DEVICES=4,5 python train_sequential_trans.py&
CUDA_VISIBLE_DEVICES=6,7 python up_transunet.py -c=up_trans_c_ridnet






