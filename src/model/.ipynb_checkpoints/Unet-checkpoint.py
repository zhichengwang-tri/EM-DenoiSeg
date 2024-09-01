import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

####SFT_GAN

# tensor_to_patches : img的channel不变,B增大,H,W成为patch_size的大小
def tensor_to_patches(tensor,patch_size):
    assert len(tensor.shape) == 4, "Input tensor should have 4 dimensions (B,C,H,W)"
    patches = tensor.unfold(2,patch_size,patch_size).unfold(3,patch_size,patch_size)
    patches = patches.permute(0,2,3,1,4,5).contiguous().view(-1,tensor.shape[1],patch_size,patch_size)
    return patches

class UNet2D(nn.Module):
    def contracting_block(self, in_channels=1, out_channels=2, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                   
                )
        return block
    
    # different out_channel for denoise and seg model
    # how to design this for seg model which outputs boundary?
    def final_block_denoise(self, in_channels, mid_channel, out_channels, kernel_size=3):
        
        block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
        )
        return block
    def final_block_seg(self, in_channels, mid_channel, out_channels, kernel_size=3):
        
        block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
        )
        return block

    def __init__(self, in_channel=1, out_channel=1, if_sigmoid=False, type = 'unified',fa=True):
        super(UNet2D, self).__init__()
        #Encode
        self.if_sigmoid= if_sigmoid
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer_denoise = self.final_block_denoise(64,32,1)
        self.final_layer_seg = self.final_block_seg(64,32,out_channel)
        self.type = type
        self.fa = fa
        self.out_sigmoid = nn.Sigmoid()

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
      
        if self.type == 'denoise':
            final_layer_denoise = self.final_layer_denoise(decode_block1)
            return final_layer_denoise
        elif self.type == 'seg':
            final_layer_seg = self.final_layer_seg(decode_block1)
            if self.if_sigmoid:
                final_layer_seg = self.out_sigmoid(final_layer_seg)
            return final_layer_seg
        elif self.type == 'seg-cons':
            final_layer_seg = self.final_layer_seg(decode_block1)
            if self.if_sigmoid:
                final_layer_seg = self.out_sigmoid(final_layer_seg)
            return bottleneck1,final_layer_seg
        else:
            final_layer_denoise = self.final_layer_denoise(decode_block1)
            final_layer_seg = self.final_layer_seg(decode_block1)
            if self.if_sigmoid:
                final_layer_seg = self.out_sigmoid(final_layer_seg)
            return  final_layer_seg,final_layer_denoise,[encode_pool1,encode_pool2,encode_pool3],[decode_block3,decode_block2,decode_block1]
        

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
      
        use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Softmax(dim=-1)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
      
class Unet_multi(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, if_sigmoid=False,cross=False):
        super(Unet_multi, self).__init__()
        #Encode
        self.if_sigmoid= if_sigmoid
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        
        self.mha2 = nn.MultiheadAttention( 64*64 ,num_heads=4,batch_first=True)
        self.mha3 = nn.MultiheadAttention( 32*32 ,num_heads=4,batch_first=True)
        self.aff_conv1 = nn.Sequential(
          self.contracting_block(in_channels=10, out_channels=32),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv2 = nn.Sequential(
          self.contracting_block(in_channels=32, out_channels=64),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv3 = nn.Sequential(
            self.contracting_block(in_channels=64, out_channels=128),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.relu = nn.ReLU()
        
        
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)
        self.cross = cross
        # fushion 
        
        self.proj = nn.Linear(128,64)
         #定义multiheadAttention参数
        d_k = 64
        num_heads = 8
        self.patch_size = 41 
        #定义线性层，将注意力输出的通道维度映射回128
        self.proj = nn.Linear(mid_channel,d_k)
        self.proj_back = nn.Linear(d_k,mid_channel)
        self.out_sigmoid = nn.Sigmoid()
        
        
    def contracting_block(self, in_channels=1, out_channels=2, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
                    # torch.nn.BatchNorm2d(out_channels),
                    # torch.nn.ReLU()
                )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
        
       
    def forward(self, x, affinity):  
        if self.cross==False:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1)
            affinity1 = self.aff_conv1(affinity)
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2) 
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2) #[b,embed,h*w]
            cross2,_ = self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)
            cross2 = cross2.reshape([b,c,h,w])  
            encode_block3 = self.conv_encode3(self.relu(cross2)+encode_pool2)
            encode_pool3 = self.conv_maxpool3(encode_block3)
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3,_ = self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)
            cross3 = cross3.reshape([b,c,h,w])  
        # Bottleneck
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool3)  
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3) #[4,256,64,64]
            cat_layer2 = self.conv_decode3(decode_block3)#[4,64,128,128]
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2) #(4,128,128,128)
            cat_layer1 = self.conv_decode2(decode_block2) #(4,32,256,256)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)#(4,64,256,256)
            final_layer = self.final_layer(decode_block1) #(4,1,256,256)
            return  final_layer 
        else:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1) 
            # aff_conv1: (b,10,256,256) --->(b,32,128,128)
            affinity1 = self.aff_conv1(affinity)
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2)
            cross2 = torch.add( self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)[1], self.mha2(encode_pool2_flatten,affinity2_flatten,affinity2_flatten)[1] ) / 2
            cross2 = cross2.reshape([b,c,h,w])  
            #or
            encode_block3 = self.conv_encode3( self.relu(cross2)+encode_pool2 )
            encode_pool3 = self.conv_maxpool3(encode_block3)
            # fushion after encoder before bottleneck
            # encode_pool3.shape: [b,128,32,32]
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3 = torch.add (self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)[1],
            self.mha3(encode_pool3_flatten,affinity3_flatten,affinity3_flatten)[1] )/2
            cross3 = cross3.reshape([b,c,h,w])  
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool2)
            
            # (b,128,64*64)
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3 )
            cat_layer2 = self.conv_decode3(decode_block3)
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
            cat_layer1 = self.conv_decode2(decode_block2)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
            final_layer = self.final_layer(decode_block1)   
            if self.if_sigmoid:
                final_layer = self.out_sigmoid(final_layer)
            return  final_layer


    

class Unet_multi_d(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, if_sigmoid=False,cross=True):
        super(Unet_multi_d, self).__init__()
        #Encode
        self.if_sigmoid= if_sigmoid
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        
        self.mha1 = nn.MultiheadAttention( 128*128,num_heads=8,batch_first=True)
        self.mha2 = nn.MultiheadAttention( 64*64,num_heads=8,batch_first=True)
        self.mha3 = nn.MultiheadAttention( 32*32,num_heads=8,batch_first=True)
        self.mha4 = nn.MultiheadAttention( 128*128,num_heads=8,batch_first=True)
        self.mha5 = nn.MultiheadAttention( 64*64,num_heads=8,batch_first=True)
        self.mha6 = nn.MultiheadAttention( 32*32,num_heads=8,batch_first=True)
        self.aff_conv1 = nn.Sequential(
          self.contracting_block(in_channels=10, out_channels=32),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv2 = nn.Sequential(
          self.contracting_block(in_channels=32, out_channels=64),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv3 = nn.Sequential(
            self.contracting_block(in_channels=64, out_channels=128),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.relu = nn.ReLU()
        
        
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)
        self.cross = cross
        # fushion 
        
        self.proj = nn.Linear(128,64)
         #定义multiheadAttention参数
        d_k = 64
        num_heads = 8
        self.patch_size = 41 
        #定义线性层，将注意力输出的通道维度映射回128
        self.proj = nn.Linear(mid_channel,d_k)
        self.proj_back = nn.Linear(d_k,mid_channel)
        self.out_sigmoid = nn.Sigmoid()
        
        #初始化multiheadAttention模型
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_k,num_heads=num_heads,batch_first=True)
        
        self.qkv_proj = nn.Linear(128, 128 * 3)
        self.attn = nn.MultiheadAttention(128, num_heads)
        self.out_proj = nn.Linear(128,128)
        
    def contracting_block(self, in_channels=1, out_channels=2, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
                    # torch.nn.BatchNorm2d(out_channels),
                    # torch.nn.ReLU()
                )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
        
       
    def forward(self, x, affinity):  
        if self.cross==False:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1)
            affinity1 = self.aff_conv1(affinity)
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2)
            cross2,_ = self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)
            cross2 = cross2.reshape([b,c,h,w])  
            encode_block3 = self.conv_encode3(self.relu(cross2)+encode_pool2)
            encode_pool3 = self.conv_maxpool3(encode_block3)
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3,_ = self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)
            cross3 = cross3.reshape([b,c,h,w])  
        # Bottleneck
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool3)
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3) 
            b,c,h,w = decode_block3.shape
            decode_block3_flatten = torch.flatten(decode_block3)
            cros1,_ = self.mha3(affinity3_flatten,decode_block3_flatten,decode_block3_flatten)
            cat_layer2 = self.conv_decode3(self.relu(cros1)+decode_block3)
            
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
            cat_layer1 = self.conv_decode2(decode_block2)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
            final_layer = self.final_layer(decode_block1)
    
            # if self.if_sigmoid:
            #     final_layer = self.out_sigmoid(final_layer)
            return  final_layer 
        else:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1) 
            # aff_conv1: (b,10,256,256) --->(b,32,128,128)
            affinity1 = self.aff_conv1(affinity)
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2)
            cross2 = torch.add( self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)[1], self.mha2(encode_pool2_flatten,affinity2_flatten,affinity2_flatten)[1] ) / 2
            cross2 = cross2.reshape([b,c,h,w])  
            #or
            encode_block3 = self.conv_encode3( self.relu(cross2)+encode_pool2 )
            encode_pool3 = self.conv_maxpool3(encode_block3)
            # fushion after encoder before bottleneck
            # encode_pool3.shape: [b,128,32,32]
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3 = torch.add (self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)[1],
            self.mha3(encode_pool3_flatten,affinity3_flatten,affinity3_flatten)[1] )/2
            cross3 = cross3.reshape([b,c,h,w])  
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool2)
            
            # (b,128,64*64)
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3 )
            cat_layer2 = self.conv_decode3(decode_block3)
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
            cat_layer1 = self.conv_decode2(decode_block2)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
            final_layer = self.final_layer(decode_block1)   
            if self.if_sigmoid:
                final_layer = self.out_sigmoid(final_layer)
            return  final_layer

class Unet_multi_light(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, if_sigmoid=False,cross=False):
        super(Unet_multi_light, self).__init__()
        #Encode
        self.if_sigmoid= if_sigmoid
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        
        self.mha1 = nn.MultiheadAttention( 64*64,num_heads=4,batch_first=True)
        self.mha2 = nn.MultiheadAttention( 32*32,num_heads=4,batch_first=True)
        self.mha3 = nn.MultiheadAttention( 16*16,num_heads=4,batch_first=True)
        self.aff_conv1 = nn.Sequential(
          self.contracting_block(in_channels=10, out_channels=10),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv2 = nn.Sequential(
          self.contracting_block(in_channels=10, out_channels=10),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv3 = nn.Sequential(
            self.contracting_block(in_channels=10, out_channels=10),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.relu = nn.ReLU()
        
        self.reduc1 = nn.Sequential(
          self.contracting_block(in_channels=64, out_channels=10),
        )
        self.enhanc1 = nn.Sequential(
          self.contracting_block(in_channels=10, out_channels=64),
        )
        self.reduc2 = nn.Sequential(
          self.contracting_block(in_channels=128, out_channels=10),
        )
        self.enhanc2 = nn.Sequential(
          self.contracting_block(in_channels=10, out_channels=128),
        )
        
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)
        self.cross = cross
        # fushion 
        
        self.proj = nn.Linear(128,64)
         #定义multiheadAttention参数
        d_k = 64
        num_heads = 8
        self.patch_size = 41 
        #定义线性层，将注意力输出的通道维度映射回128
        self.proj = nn.Linear(mid_channel,d_k)
        self.proj_back = nn.Linear(d_k,mid_channel)
        self.out_sigmoid = nn.Sigmoid()
        
        #初始化multiheadAttention模型
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_k,num_heads=num_heads,batch_first=True)
        
        self.qkv_proj = nn.Linear(128, 128 * 3)
        self.attn = nn.MultiheadAttention(128, num_heads)
        self.out_proj = nn.Linear(128,128)
        
    def contracting_block(self, in_channels=1, out_channels=2, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
                    # torch.nn.BatchNorm2d(out_channels),
                    # torch.nn.ReLU()
                )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
        
       
    def forward(self, x, affinity):  
        if self.cross==False:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1) #(32,128,128)
            affinity1 = self.aff_conv1(affinity) 
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2) #(64,64,64)
            encode_pool2_l = self.reduc1(encode_pool2)   #(10,64,64)
            affinity2 = self.aff_conv2(affinity1) #(10,64,64)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2)
            encode_pool2_flatten = torch.flatten(encode_pool2_l,start_dim=2)
            cross2,_ = self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)
            cross2 = cross2.reshape([b,c,h,w]) 
            cross2 = self.enhanc1(cross2)
            encode_block3 = self.conv_encode3(self.relu(cross2)+encode_pool2)
            encode_pool3 = self.conv_maxpool3(encode_block3) #(128,32,32)
            encode_pool3_l = self.reduc2(encode_pool3)
            affinity3 = self.aff_conv3(affinity2) 
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3_l,start_dim=2)
            cross3,_ = self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)
            cross3 = cross3.reshape([b,c,h,w])  
            cross3 = self.enhanc2(cross3)
        # Bottleneck
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool3)  
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3) #[4,256,64,64]
            cat_layer2 = self.conv_decode3(decode_block3)#[4,64,128,128]
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2) #(4,128,128,128)
            cat_layer1 = self.conv_decode2(decode_block2) #(4,32,256,256)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)#(4,64,256,256)
            final_layer = self.final_layer(decode_block1) #(4,1,256,256)
            return  final_layer 
        else:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1) 
            # aff_conv1: (b,10,256,256) --->(b,32,128,128)
            affinity1 = self.aff_conv1(affinity)
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2)
            cross2 = torch.add( self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)[1], self.mha2(encode_pool2_flatten,affinity2_flatten,affinity2_flatten)[1] ) / 2
            cross2 = cross2.reshape([b,c,h,w])  
            #or
            encode_block3 = self.conv_encode3( self.relu(cross2)+encode_pool2 )
            encode_pool3 = self.conv_maxpool3(encode_block3)
            # fushion after encoder before bottleneck
            # encode_pool3.shape: [b,128,32,32]
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3 = torch.add (self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)[1],
            self.mha3(encode_pool3_flatten,affinity3_flatten,affinity3_flatten)[1] )/2
            cross3 = cross3.reshape([b,c,h,w])  
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool2)
            
            # (b,128,64*64)
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3 )
            cat_layer2 = self.conv_decode3(decode_block3)
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
            cat_layer1 = self.conv_decode2(decode_block2)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
            final_layer = self.final_layer(decode_block1)   
            if self.if_sigmoid:
                final_layer = self.out_sigmoid(final_layer)
            return  final_layer
    


class UNet_self(nn.Module):
    def contracting_block(self, in_channels=1, out_channels=2, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
                    # torch.nn.BatchNorm2d(out_channels),
                    # torch.nn.ReLU()
                )
        return block

    def __init__(self, in_channel=1, out_channel=1, if_relu=False,if_sigmoid=False,):
        super(UNet_self, self).__init__()
        #Encode
        self.if_sigmoid= if_sigmoid
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.if_relu = if_relu
        # Bottleneck
        mid_channel = 128
        self.mha1 = nn.MultiheadAttention( 128*128,num_heads=8,batch_first=True)
        self.mha2 = nn.MultiheadAttention( 64*64,num_heads=8,batch_first=True)
        self.mha3 = nn.MultiheadAttention( 32*32,num_heads=8,batch_first=True)
        self.aff_conv1 = nn.Sequential(
          self.contracting_block(in_channels=2, out_channels=32),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv2 = nn.Sequential(
          self.contracting_block(in_channels=32, out_channels=64),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv3 = nn.Sequential(
            self.contracting_block(in_channels=64, out_channels=128),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)
        
        # fushion 
        
        self.proj = nn.Linear(128,64)
         #定义multiheadAttention参数
        d_k = 64
        num_heads = 8
        self.patch_size = 41 
        #定义线性层，将注意力输出的通道维度映射回128
        self.proj = nn.Linear(mid_channel,d_k)
        self.proj_back = nn.Linear(d_k,mid_channel)
        self.out_sigmoid = nn.Sigmoid()
        
        #初始化multiheadAttention模型
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_k,num_heads=num_heads,batch_first=True)
        
        self.qkv_proj = nn.Linear(128, 128 * 3)
        self.attn = nn.MultiheadAttention(128, num_heads)
        self.out_proj = nn.Linear(128,128)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
        
       
    def forward(self, x, y):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1) 
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)  
        b,c,h,w = encode_pool2.shape
        encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2)
        cross2,_ = self.mha2(encode_pool2_flatten,encode_pool2_flatten,encode_pool2_flatten)
        cross2 = cross2.reshape([b,c,h,w])  
        if self.if_relu:
            encode_block3 = self.conv_encode3( self.relu(cross2)+encode_pool2 )
        else:
            encode_block3 = self.conv_encode3( encode_pool2 )
            
        encode_pool3 = self.conv_maxpool3(encode_block3)
        b,c,h,w = encode_pool3.shape
        # fushion after encoder before bottleneck
        encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
        cross3,_ = self.mha3(encode_pool3_flatten,encode_pool3_flatten,encode_pool3_flatten)
        cross3 = cross3.reshape([b,c,h,w])  
        if self.if_relu:
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool3)
        else:
            bottleneck1 = self.bottleneck( encode_pool3 )
        # (b,128,64*64)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3 )
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)    
        if self.if_sigmoid:
            final_layer = self.out_sigmoid(final_layer)
        return  final_layer


class UNet_encoder(nn.Module):
    def __init__(self,in_channel=2,if_sigmoid=False):
        super(UNet_encoder,self).__init__()
        self.if_sigmoid= if_sigmoid
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        
    def contracting_block(self, in_channels=1, out_channels=2, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block
      
    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        return encode_pool3
        
        
        
        
class UNet_decoder(nn.Module):
    
    def __init__(self,in_channel=128,out_channel=1):
        super(UNet_decoder,self).__init__()   
        #encoder to bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)
        self.out_sigmoid = nn.Sigmoid()

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1)
                    # torch.nn.BatchNorm2d(out_channels),
                    # torch.nn.ReLU()
                )
        return block
    
    def forward(self, x):
        # Bottleneck
        bottleneck1 = self.bottleneck(x)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, x)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, x)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, x)
        final_layer = self.final_layer(decode_block1)
        if self.if_sigmoid:
            final_layer = self.out_sigmoid(final_layer)
        return  final_layer

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width, d = x.size()
        query = self.query(x).view(batch_size, -1, height * width * d).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width * d)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value(x).view(batch_size, -1, height * width * d) 
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, -1, height, width, d)
        return out + x

        
        
        
if __name__ == "__main__":
    import numpy as np
    from ptflops import get_model_complexity_info

    # model = UNet().to('cuda:0')
    model = UNet_skele().to('cuda:0')

    input = np.random.random((1, 1, 256, 256)).astype(np.float32)
    x = torch.tensor(input).to('cuda:0')
    out, skele = model(x)
    print(out.shape)
    print(skele.shape)

    # macs, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
