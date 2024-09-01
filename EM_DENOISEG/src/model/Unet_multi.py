import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

    
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
        self.mha2 = nn.MultiheadAttention( 64 ,num_heads=4)
        self.mha3 = nn.MultiheadAttention( 128 ,num_heads=4)
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
        d_k = 64
        num_heads = 8
        self.patch_size = 41 
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
            affinity2_flatten = torch.flatten(affinity2,start_dim=2).permute(2,0,1)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2).permute(2,0,1)
            cross2,_ = self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)
            cross2 = cross2.permute(1,2,0).reshape([b,c,h,w])
            encode_block3 = self.conv_encode3(self.relu(cross2)+encode_pool2)
            encode_pool3 = self.conv_maxpool3(encode_block3)
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2).permute(2,0,1)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2).permute(2,0,1)
            cross3,_ = self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)
            cross3 = cross3.permute(1,2,0).reshape([b,c,h,w])
           # Bottleneck
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool3)  
            # Decode
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3) 
            cat_layer2 = self.conv_decode3(decode_block3) 
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2) 
            cat_layer1 = self.conv_decode2(decode_block2) 
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
            final_layer = self.final_layer(decode_block1) 
            return  final_layer 
        else:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1) 
            affinity1 = self.aff_conv1(affinity)
            encode_block2 = self.conv_encode2(encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2)
            cross2 = torch.add( self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten)[1], self.mha2(encode_pool2_flatten,affinity2_flatten,affinity2_flatten)[1] ) / 2
            cross2 = cross2.reshape([b,c,h,w])  
            encode_block3 = self.conv_encode3( self.relu(cross2)+encode_pool2 )
            encode_pool3 = self.conv_maxpool3(encode_block3)
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3 = torch.add (self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)[1],
            self.mha3(encode_pool3_flatten,affinity3_flatten,affinity3_flatten)[1] )/2
            cross3 = cross3.reshape([b,c,h,w])  
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool2)
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3 )
            cat_layer2 = self.conv_decode3(decode_block3)
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
            cat_layer1 = self.conv_decode2(decode_block2)
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
            final_layer = self.final_layer(decode_block1)   
            if self.if_sigmoid:
                final_layer = self.out_sigmoid(final_layer)
            return  final_layer
        
class Unet_multi_v2(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, if_sigmoid=False,cross=False):
        super(Unet_multi_v2, self).__init__()
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
        self.mha1 = nn.MultiheadAttention( 32 ,num_heads=4,batch_first=True)
        self.mha2 = nn.MultiheadAttention( 64 ,num_heads=4,batch_first=True)
        self.mha3 = nn.MultiheadAttention( 128 ,num_heads=4,batch_first=True)
        self.mha6 = nn.MultiheadAttention( 256 ,num_heads=4,batch_first=True)
        self.mha5 = nn.MultiheadAttention( 128,num_heads=4,batch_first=True)
        self.mha4 = nn.MultiheadAttention( 64,num_heads=4,batch_first=True)
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
        self.aff_conv4 = nn.Sequential(
        self.contracting_block(in_channels=10, out_channels=64),
        )
        self.aff_conv5 = nn.Sequential(
            self.contracting_block(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.aff_conv6 = nn.Sequential(
            self.contracting_block(in_channels=128, out_channels=256),
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
            b,c,h,w = affinity1.shape  
            affinity1_flatten = torch.flatten(affinity1,start_dim=2).permute(2,0,1)
            encode_pool1_flatten = torch.flatten(encode_pool1,start_dim=2).permute(2,0,1)
            cross1,_ = self.mha1(affinity1_flatten,encode_pool1_flatten,encode_pool1_flatten) #(4096,16,64)
            cross1 = cross1.permute(1,2,0).reshape([b,c,h,w])
            encode_block2 = self.conv_encode2(self.relu(cross1)+encode_pool1)
            encode_pool2 = self.conv_maxpool2(encode_block2)
            affinity2 = self.aff_conv2(affinity1)
            b,c,h,w = affinity2.shape  
            affinity2_flatten = torch.flatten(affinity2,start_dim=2).permute(2,0,1)
            encode_pool2_flatten = torch.flatten(encode_pool2,start_dim=2).permute(2,0,1)
            cross2,_ = self.mha2(affinity2_flatten,encode_pool2_flatten,encode_pool2_flatten) #(4096,16,64)
            cross2 = cross2.permute(1,2,0).reshape([b,c,h,w])
            encode_block3 = self.conv_encode3(self.relu(cross2)+encode_pool2)
            encode_pool3 = self.conv_maxpool3(encode_block3)
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2).permute(2,0,1)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2).permute(2,0,1)
            cross3,_ = self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)
            cross3 = cross3.permute(1,2,0).reshape([b,c,h,w])
        # Bottleneck
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool3)  
            # Decode
            affinity4 = self.aff_conv4(affinity)
            affinity5 = self.aff_conv5(affinity4)
            affinity6 = self.aff_conv6(affinity5)
            decode_block3 = self.crop_and_concat(bottleneck1, encode_block3) #[4,256,64,64]
            b,c,h,w = affinity6.shape
            affinity6_flatten = torch.flatten(affinity6,start_dim=2).permute(2,0,1)
            decode_block3_flatten = torch.flatten(decode_block3,start_dim=2).permute(2,0,1)
            cross6,_ = self.mha6(affinity6_flatten,decode_block3_flatten,decode_block3_flatten)
            cross6 = cross6.permute(1,2,0).reshape([b,c,h,w])
            cat_layer2 = self.conv_decode3(self.relu(cross6)+decode_block3)
            decode_block2 = self.crop_and_concat(cat_layer2, encode_block2) 
            b,c,h,w = affinity5.shape
            affinity5_flatten = torch.flatten(affinity5,start_dim=2).permute(2,0,1)
            decode_block2_flatten = torch.flatten(decode_block2,start_dim=2).permute(2,0,1)
            cross5,_ = self.mha5(affinity5_flatten,decode_block2_flatten,decode_block2_flatten)
            cross5 = cross5.permute(1,2,0).reshape([b,c,h,w])
            cat_layer1 = self.conv_decode2(self.relu(cross5)+decode_block2) 
            decode_block1 = self.crop_and_concat(cat_layer1, encode_block1) 
            b,c,h,w = affinity4.shape
            affinity4_flatten = torch.flatten(affinity4,start_dim=2).permute(2,0,1)
            decode_block1_flatten = torch.flatten(decode_block1,start_dim=2).permute(2,0,1)
            cross4,_ = self.mha4(affinity4_flatten,decode_block1_flatten,decode_block1_flatten)
            cross4 = cross4.permute(1,2,0).reshape([b,c,h,w])
            final_layer = self.final_layer(self.relu(cross4)+decode_block1) #(4,1,256,256)
            return  final_layer 
        else:
            encode_block1 = self.conv_encode1(x)
            encode_pool1 = self.conv_maxpool1(encode_block1) 
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
            affinity3 = self.aff_conv3(affinity2)
            b,c,h,w = affinity3.shape
            affinity3_flatten = torch.flatten(affinity3,start_dim=2)
            encode_pool3_flatten = torch.flatten(encode_pool3,start_dim=2)
            cross3 = torch.add (self.mha3(affinity3_flatten,encode_pool3_flatten,encode_pool3_flatten)[1],
            self.mha3(encode_pool3_flatten,affinity3_flatten,affinity3_flatten)[1] )/2
            cross3 = cross3.reshape([b,c,h,w])  
            bottleneck1 = self.bottleneck(self.relu(cross3)+encode_pool2)
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