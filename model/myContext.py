import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

import os
import math

channel_context = 40

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(in_planes)
        #self.norm3 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, in_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        #self.convonek = nn.Conv2d(planes, in_planes, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        # y = self.relu(self.norm3(self.convonek(y)))


        #if self.downsample is not None:
        #    x = self.downsample(x)

        return self.relu(x+y)
    

class FlowContext(nn.Module):
    def __init__(self, in_planes=3, planes=16, stride=1):
        super().__init__()
        self.Before_context = nn.Sequential(
            conv(in_planes=in_planes, out_planes=channel_context, kernel_size=3, padding=1, stride=2),
            conv(in_planes=channel_context, out_planes=channel_context),
            conv(in_planes=channel_context, out_planes=channel_context),
            conv(in_planes=channel_context, out_planes=channel_context),
            deconv(in_planes=channel_context, out_planes=channel_context//2)
        )
        self.After_context = nn.Sequential(
            conv(in_planes=in_planes, out_planes=channel_context, kernel_size=3, padding=1, stride=2),
            conv(in_planes=channel_context, out_planes=channel_context),
            conv(in_planes=channel_context, out_planes=channel_context),
            conv(in_planes=channel_context, out_planes=channel_context),
            deconv(in_planes=channel_context, out_planes=channel_context//2)
        )
        '''
        self.extractblock = nn.Sequential(
            ResidualBlock(in_planes=2*3, planes=4*3), # 311
            ResidualBlock(in_planes=2*2*3, planes=2*4*3),
            ResidualBlock(in_planes=2*4*3, planes=6)
        )
        '''
        self.extractblock = nn.Sequential(
            conv(6,12),
            conv(12,12),
            conv(12,6)
        )

        # self.pooling # Coming
        self.melt_conv1 = nn.Sequential(
            conv(channel_context//2 + 6 + 3,channel_context//2, kernel_size=3, padding=1, stride=stride),
            conv(channel_context//2, channel_context//2),
            conv(channel_context//2, channel_context//2)
            #conv(channel_context//2, 3)
        )
        self.melt_conv2 = nn.Sequential(
            conv(channel_context//2 + 6 + 3,channel_context//2, kernel_size=3, padding=1, stride=stride),
            conv(channel_context//2, channel_context//2),
            conv(channel_context//2, channel_context//2)
            # conv(channel_context//2, 3)
        )
        self.melt_conv3 = nn.Sequential(
            conv(channel_context, channel_context//2),
            conv(channel_context//2, channel_context//4),
            conv(channel_context//4, 3)
        )


    
    
    # def forward(self, flow0, flow1, image0, image1):
    def forward(self, image0, image1, warped_img0, warped_img1, mask, flow):
        flow1 = flow[:, :2]
        flow2 = flow[2:,:]
        image_concat = torch.concat([image0, image1], dim=1) # B*6*H*W
        fea_extrcat = self.extractblock(image_concat)
        fea_image0 = self.Before_context(warped_img0)
        fea_image1 = self.After_context(warped_img1)

        melt1 = self.melt_conv1(torch.concat((warped_img0, fea_extrcat,fea_image0),dim=1))
        melt2 = self.melt_conv2(torch.concat((warped_img0, fea_extrcat,fea_image1),dim=1))
        melt_final = self.melt_conv3(torch.concat((melt1,melt2),dim=1))
        

        return melt_final


        

 