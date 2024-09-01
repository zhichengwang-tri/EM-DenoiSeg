import torch 
import time
import torch.nn as nn

from .Unet import UNet2D
from .Unet import  sft , UNet_fushion_feature,Unet_multi,map_Unet_multi,UNet_self
from .adaptive import adaptive_UNet , adaptive_UNet_attention
from .fushion import UNet_fushion_multi,DnCNN_fushion_multi,UNet_fushion_multi_cheap
from .new import res_encoder, Decoder_seg, Decoder_denoise,UNet_encoder,SharedEncoder, UNet_decoder_seg,UNet_decoder_denoise,UNet_guidance,UNet_guidance2
import src.model.architectures as arch
from src.model.unet2d_residual import ResidualUNet2D_dn
from torchvision.models import resnet50
def build_model_seg(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
    if cfg.MODEL.MODEL_seg.model_type == 'unet2d':      
            if cfg.TRAIN.multi_scale:
                print('load unet2d multi-scale model!')
                model_seg =UNet_ms(in_channel=cfg.MODEL.MODEL_seg.input_nc, out_channel=cfg.MODEL.MODEL_seg.output_nc).to(device)
            else: 
                print('load unet2d model!')
                model_seg = UNet2D(in_channel=cfg.MODEL.MODEL_seg.input_nc, out_channel=cfg.MODEL.MODEL_seg.output_nc).to(device)
        
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_seg= nn.DataParallel(model_seg)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_seg.to(device)
   
def build_model_encoder(cfg,writer):
    print("Building model on",end='',flush=True)
    t1=time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.encoder.model_type == 'res':
        print('load res for encoder')
        # num_classes参数应该怎么设置
        model = resnet50(pretrained=True)
        model.conv1=torch.nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        encoder = torch.nn.Sequential(*list(model.children())[:-3]).to(device)   
    else:
        encoder = UNet_encoder()
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            encoder = nn.DataParallel(encoder)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return encoder.to(device)
        

def build_model_decoder(cfg,writer):
    print("Building model on",end='',flush=True)
    t1=time.time()
    device = torch.device('cuda:0')
    #暂未选定decoder类型
    print('load decoder for denoise&segmentation ')
    if cfg.MODEL.encoder.model_type == 'unet':
        decoder_seg = UNet_decoder_seg().to(device)
        decoder_denoise =UNet_decoder_denoise().to(device)
    else:
        decoder_seg = Decoder_seg(num_classes=cfg.MODEL.decoder.seg.output_nc).to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            decoder_denoise= nn.DataParallel(decoder_denoise)
            decoder_seg = nn.DataParallel(decoder_seg)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return decoder_seg.to(device),decoder_denoise.to(device)


def build_model_guidance(cfg,writer,type):
    print("Building model on",end='',flush=True)
    t1=time.time()
    device = torch.device('cuda:0')
    #暂未选定decoder类型
    print('load decoder for guidance-based denoising decoder ')
    if type=='guidance1':
        model_denoise = UNet_guidance().to(device)
    else:
        model_denoise = UNet_guidance2().to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_denoise= nn.DataParallel(model_denoise)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_denoise.to(device)  

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
    if cfg.MODEL.MODEL_seg.model_type == 'unet':     
        print('load unet2d model!')
        model_seg = UNet2D(in_channel=cfg.MODEL.MODEL_seg.input_nc, out_channel=cfg.MODEL.MODEL_seg.output_nc).to(device)

    if cfg.MODEL.MODEL_denoise.model_type == 'DnCNN2d':
        print('load dncnn model!')
        model_denoise = DnCNN(channels=1).to(device)  
    elif cfg.MODEL.MODEL_denoise.model_type == 'residual':
        model_denoise = ResidualUNet2D_dn(in_channels=cfg.MODEL.input_nc,
                                out_channels=cfg.MODEL.output_nc,
                                nfeatures=cfg.MODEL.filters).to(device)
        
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_seg= nn.DataParallel(model_seg)
            model_denoise = nn.DataParallel(model_denoise)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_seg.to(device), model_denoise.to(device)

def build_model_fushion(cfg,writer,in_channel,model_type,cross,relu=False):
    print('Building fushion model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
   
    #attention only:
    if model_type== 'multi':   
        model_fushion = Unet_multi(in_channel=in_channel,cross=cross)
    #attention only:
    elif model_type == 'adaptive':
        model_fushion = adaptive_UNet(in_channel=in_channel) #目前adaptive模块遇到多GPU情况下有问题
    elif model_type == 'self':
        model_fushion = UNet_self(in_channel=in_channel,if_relu=relu) #目前adaptive模块遇到多GPU情况下有问题
    #adaptive + attention
    elif model_type == 'adaptive_multi':
        model_fushion = adaptive_UNet_attention(in_channel=in_channel)
    
    elif  model_type == 'fushion':      
        model_fushion = UNet_fushion(in_channel=2).to(device)
    elif model_type == 'map':
        model_fushion = map_Unet_multi().to(device)
    else:
        model_fushion = arch.SFT_Net_torch()    
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            
            # model_fushion= nn.DataParallel(model_fushion)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    
    return model_fushion.to(device) 
   

def build_model_adaptive(cfg,writer):
    print('Building fushion model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
    if cfg.MODEL.MODEL_fushion.model_type == 'unet2d':      
        model_fushion = adaptive_UNet_fushion().to(device)   
    if cfg.MODEL.MODEL_fushion.model_type == 'DnCNN2d':
        model_fushion = DnCNN_fushion_multi().to(device)    
        print('load dncnn model!')
    cuda_count = torch.cuda.device_count() 
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_fushion= nn.DataParallel(model_fushion)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_fushion.to(device) 

'''
build_model_fushion_feature指的是将affinity在encoder后的feature层面送进融合网络
'''
def build_model_fushion_feature(cfg,writer):
    print('Building fushion model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
    if cfg.MODEL.MODEL_fushion.model_type == 'unet2d':      
        model_fushion = UNet_fushion_feature().to(device)
    if cfg.MODEL.MODEL_fushion.model_type == 'DnCNN2d':
        print('load dncnn model!')
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_fushion= nn.DataParallel(model_fushion)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_fushion.to(device)     

def build_model_sft(cfg,writer):
    print('Building fushion model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
    if cfg.MODEL.MODEL_fushion.model_type == 'unet2d':      
        model_sft = UNet_sft().to(device)
    
    if cfg.MODEL.MODEL_fushion.model_type == 'DnCNN2d':
            print('load dncnn model!')
    cuda_count = torch.cuda.device_count() 
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_sft= nn.DataParallel(model_sft)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    
    return model_sft.to(device) 

def build_model_fushion_multi(cfg,writer):
    print('Building fushion model on ', end='', flush=True)
    t1 = time.time() 
    device = torch.device('cuda:0')
    # try to  change device to multiple-gpus according to cfg
    if cfg.MODEL.MODEL_fushion.model_type == 'unet2d':      
        model_fushion = UNet_fushion_multi_cheap().to(device)
    if cfg.MODEL.MODEL_fushion.model_type == 'DnCNN2d':
        # model_fushion = DnCNN_fushion_multi().to(device)
        print('load dncnn model!')
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_fushion= nn.DataParallel(model_fushion)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model_fushion.to(device)     
