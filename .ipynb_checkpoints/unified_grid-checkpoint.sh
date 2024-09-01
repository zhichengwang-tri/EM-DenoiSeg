#!/bin/bash
cd /code/JDAS_2stage/jdas_full/ && pip install albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple \

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_unified.py -c=unified_a_4&
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_unified.py -c=unified_a_2
