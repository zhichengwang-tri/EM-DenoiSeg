cd /code/JDAS_2stage/jdas_full/ && pip install ml_collections albumentations torch==1.4.0 torchvision==0.5.0  -i https://pypi.tuna.tsinghua.edu.cn/simple \

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_unified_trans.py -c=unified_v2_c_trans&
CUDA_VISIBLE_DEVICES=4,5,6,7 python up_transunet.py -c=up_c_transunet
