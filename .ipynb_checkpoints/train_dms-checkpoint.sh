#!/bin/bash
cd /code/JDAS_2stage/jdas_full/ && pip install albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple \

CUDA_VISIBLE_DEVICES=0,1 python DMS.py -c=dms_c_0.5&
CUDA_VISIBLE_DEVICES=2,3 python DMS.py -c=dms_ac34_0.5&
CUDA_VISIBLE_DEVICES=4,5 python DMS.py -c=dms_b_0.1&
CUDA_VISIBLE_DEVICES=6,7 python DMS.py -c=dms_b_0.15





