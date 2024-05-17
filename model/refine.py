import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
cu = 32
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, cu)
        self.conv2 = Conv2(cu, 2*cu)
        self.conv3 = Conv2(2*cu, 4*cu)
        self.conv4 = Conv2(4*cu, 8*cu)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]

class downContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputContext0 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding =1)
        self.inputContext1 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding =1)
        self.inputContextfusion = nn.Conv2d(8+3, 16, 3, 1, 1)

    def forward(self, context, origin_image):
        context_d = self.inputContext0(context)
        context_dd = self.inputContext1(context_d)
        out = self.inputContextfusion(torch.cat([origin_image, context_dd], dim = 1))
        return out 
    # out: 16

c=16
class OriUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = Conv2(17, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)


class Unet(nn.Module):
    def __init__(self, hidden_dim=32, shift_dim=32):
        # orward_shiftedFeature, backward_shftedFeature, forwardContext, backwardContext
        super().__init__()
        self.hidden_dim=hidden_dim
        self.shft_dim = shift_dim
        self.hs_dim = self.hidden_dim + self.shft_dim # 64
        self.down0 = Conv2(in_planes=(self.hs_dim), out_planes=(self.hs_dim))
        self.down1 = Conv2((self.hs_dim)*2, (self.hs_dim)*2)
        self.down2 = Conv2((self.hs_dim)*4, (self.hs_dim)*4)
        self.down3 = Conv2((self.hs_dim)*4, (self.hs_dim)*4)
        self.up0 = deconv((self.hs_dim)*4, (self.hs_dim)*4)
        self.up1 = deconv((self.hs_dim)*8, (self.hs_dim)*2)
        self.up2 = deconv((self.hs_dim)*4, (self.hs_dim)*1)
        self.up3 = deconv((self.hs_dim)*2, (self.hs_dim)*2)
        self.conv = nn.Conv2d((self.hs_dim)*2, 3, 3, 1, 1)

    def forward(self, forward_shiftedFeature, backward_shiftedFeature, forwardContext_d2, forwardContext_d4, backwardContext_d2, backwardContext_d4):
        # forward_shiftedFeature, backward_shiftedFeature, forwardContext_d2, forwardContext_d4, backwardContext_d2, backwardContext_d4)
        s0 = self.down0(torch.cat([forward_shiftedFeature, backward_shiftedFeature], dim=1)) # 32+32
        s1 = self.down1(torch.cat([s0, forwardContext_d2, backwardContext_d2], dim=1 )) # 64 + (32+32)
        s2 = self.down2(torch.cat([s1, forwardContext_d4, backwardContext_d4], dim=1)) # 128 + (64+64)
        s3 = self.down3(s2)
        x = self.up0(s3)
        x = self.up1(torch.cat([x, s2], 1)) 
        x = self.up2(torch.cat([x, s1], 1)) 
        x = self.up3(torch.cat([x, s0], 1)) 
        x = self.conv(x)
        
        return torch.sigmoid(x)
