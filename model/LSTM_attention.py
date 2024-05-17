import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import model.laplacian as modelLap
from model.warplayer import warp
from model.refine import *
from model.myContext import *
from model.loss import *
from model.myLossset import *
from model.Pyramid import FeaturePyramid as FPyramid
# Attention test
import model.Attenions as att

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = 48



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock0 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat        
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask
       
          

class PyramidFBwardExtractor(nn.Module):
    # input 3 outoput 6
    def __init__(self, in_plane=3, c=32, att_mode='se'):
        super().__init__()
        # set stride = 2 to downsample
        # as for 224*224, the current output shape is 112*112
        self.c = c
        self.out_plane = self.c*2
        self.in_plane = in_plane
        self.pyramid = FPyramid(self.c) # pyramid d1: c*h/2*w/2, ; pyramid 2: 2c * h/4* d/4
        self.fromimage = conv(self.in_plane, self.c, kernel_size=3, stride=1, padding=1)
        self.downsample = conv(self.c, 2*self.c, kernel_size=3, stride=2, padding=1)

        self.forwardFeatureList = []
        # self.upsample_d2 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.out_plane, kernel_size=3, stride=2, padding=1, output_padding=1) # H/2 -> H
        # self.upsample_d4 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.out_plane, kernel_size=3, stride=4, padding=1, output_padding=1) # H/4 -> H
        
        
        if att_mode == 'se':
            self.attention_d2 = att.SELayer(channel=self.out_plane//2, reduction=16,pool_mode='avg')
            self.attention_d4 = att.SELayer(channel=self.out_plane, reduction=16,pool_mode='avg')
        elif att_mode == 'cbam':
            self.attention = att.CBAM(in_channel=self.out_plane, ratio=4, kernel_size=7)
        elif att_mode == 'none':
            self.attention= nn.Sequential()

    def forward(self, allframes):
        # all frames: B*21*H*W  --> 
        # x is concated frames [0,2,4,6] -> [(4*3),112,112] 
        forwardFeatureList_d2 = []
        forwardFeatureList_d4 = []
        for i in range(0,4):
            x = allframes[:, 6*i:6*i+3].clone()
            y_d2, y_d4 = self.pyramid(x)
            
            # x = self.downsample(y)  # 224*224 -> 112*112
            # x = self.conv0(x)       # Pass through conv layers
            
            # Upsample and add to the original y tensor
            
            x_d2 = self.attention_d2(y_d2)
            
            x_d4 = self.attention_d4(y_d4)
            
            
            forwardFeatureList_d2.append(x_d2)
            forwardFeatureList_d4.append(x_d4)
            # self.forwardFeatureList.append(x)


        return forwardFeatureList_d2, forwardFeatureList_d4
    # Output: BNCHW

class unitConvGRU(nn.Module):
    # Formula:
    # I_t = Sigmoid(Conv(x_t;W_{xi}) + Conv(h_{t-1};W_{hi}) + b_i)
    # F_t = Sigmoid(Conv(x_t;W_{xf}) + Conv(h_{t-1};W_{hi}) + b_i)
    def __init__(self, hidden_dim=64, input_dim=c):
        # 192 = 4*4*12  
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h
  
# c = 48
class ConvGRUFeatures(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # current encoder: all frames ==> all features 
        self.img2Fencoder = PyramidFBwardExtractor()
        self.img2Bencoder = PyramidFBwardExtractor()
        self.hidden_dim_d2 = hidden_dim
        self.hidden_dim_d4 = hidden_dim*2
        self.forwardgru_d2 = unitConvGRU(hidden_dim=self.hidden_dim_d2, input_dim=hidden_dim)
        self.backwardgru_d2 = unitConvGRU(hidden_dim=self.hidden_dim_d2, input_dim=hidden_dim)
        self.forwardgru_d4 = unitConvGRU(hidden_dim=self.hidden_dim_d4, input_dim=2*hidden_dim)
        self.backwardgru_d4 = unitConvGRU(hidden_dim=self.hidden_dim_d4, input_dim=2*hidden_dim)

        


    def forward(self, allframes):
        # aframes = allframes_N.view(b,n*c,h,w)
        # Output: BNCHW
        fcontextlist_d2 = [] # c = 32
        bcontextlist_d2 = [] # c = 32
        fcontextlist_d4 = [] # 2c = 64
        bcontextlist_d4 = [] # 2c = 64
        fallfeatures_d2, fallfeatures_d4 = self.img2Fencoder(allframes)
        ballfeatures_d2, ballfeatures_d4 = self.img2Bencoder(allframes)
        b, _, h, w = allframes.size()


        # forward GRU 
        # h' = gru(h,x)
        # Method A: zero initialize Hiddenlayer
        forward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h//2, w//2),device=device )
        backward_hidden_initial_d2 = torch.zeros((b, self.hidden_dim_d2, h//2, w//2), device=device)
        forward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h//4, w//4),device=device )
        backward_hidden_initial_d4 = torch.zeros((b, self.hidden_dim_d4, h//4, w//4), device=device)
        # n=4
        # I skipped the 0 -> first image
        for i in range(0,4):
            if i == 0:
                fhidden = self.forwardgru_d2(forward_hidden_initial_d2, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(backward_hidden_initial_d2, ballfeatures_d2[-i-1])
            else:
                fhidden = self.forwardgru_d2(fhidden, fallfeatures_d2[i])
                bhidden = self.backwardgru_d2(bhidden, ballfeatures_d2[-i-1])
                fcontextlist_d2.append(fhidden)
                bcontextlist_d2.append(bhidden)
        for i in range(0,4):
            if i == 0:
                fhidden = self.forwardgru_d4(forward_hidden_initial_d4, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(backward_hidden_initial_d4, ballfeatures_d4[-i-1])
            else:
                fhidden = self.forwardgru_d4(fhidden, fallfeatures_d4[i])
                bhidden = self.backwardgru_d4(bhidden, ballfeatures_d4[-i-1])
                fcontextlist_d4.append(fhidden)
                bcontextlist_d4.append(bhidden)
        return fcontextlist_d2, fcontextlist_d4, bcontextlist_d2, bcontextlist_d4
        # return forwardFeature, backwardFeature
        # Now iterate through septuplet and get three inter frames


class SingleImageExtractor(nn.Module):
    def __init__(self, in_plane=3, out_plane=32, att_mode='se'):
        super().__init__()
        # set stride = 2 to downsample
        # as for 224*224, the current output shape is 112*112
        self.out_plane = out_plane
        self.fromimage = conv(in_plane, 32, kernel_size=3, stride=1, padding=1)
        self.downsample = conv(32, 2*32, kernel_size=3, stride=1, padding=1)

        self.conv0 = nn.Sequential(
            conv(2*32, 2*32, 3, 1, 1),
            conv(2*32, 4*32, 3, 1, 1),
            conv(4*32, 4*32, 3, 1, 1),
            conv(4*32, self.out_plane, 3, 1, 1),
            )
       

    def forward(self, single):
        s = self.fromimage(single)
        s = self.downsample(s) # not downsample for now
        s = self.conv0(s)
        s = torch.tanh(s)
        return s
    
class Loaded_Modified_IFNet(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        '''
        self.block0 = IFBlock(6+1, c=240)
        self.block1 = IFBlock(13+4+1, c=150)
        self.block2 = IFBlock(13+4+1, c=90)
        self.block_tea = IFBlock(16+4+1, c=90)
        self.contextnet = Contextnet()
        '''
        # Notice that the training start at Contextnet

        self.block0 = pretrained_model.block0
        self.block1 = pretrained_model.block1
        self.block2 = pretrained_model.block2
        self.block_tea = pretrained_model.block_tea
        self.InterpolationEncoder = SingleImageExtractor()
        
        # self.contextnet = pretrained_model.contextnet()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def forward(self, x, scale_list=[4,2,1]):
        # forwardContext/backwardContext is forwardFeature[i], only pick up the one for the current interpolation
        # final_merged, loss = self.mIFnetframesset[i], for(wardFeature[3*i], backwardFeature[3*i+2])
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        # stdv = np.random.uniform(0.0, 5.0)
        # img0 = (img0 + stdv * torch.randn(*img0.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp
        # img1 = (img1 + stdv * torch.randn(*img1.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp       

        loss_ssimd = 0
        merged_features = []
        # eps = 1e-8
# ----------------

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            # merged.append((warped_img0, warped_img1))
        interframe_f0 = self.InterpolationEncoder(img0)
        interframe_f0 = warp(interframe_f0, flow[:,:2])
        merged_features.append(interframe_f0)
        interframe_f1 = self.InterpolationEncoder(img1)
        interframe_f1 = warp(interframe_f1, flow[:,2:4])
        merged_features.append(interframe_f1)
        
        return merged_features
# ----------------

class Unetdecoder(nn.Module):
    def __init__(self):
        self.decoder = nn.Sequential()
        
    def forward(self, x):
        return self.decoder(x)

class newMergeIFnet(nn.Module):
    def __init__(self, pretrained_model, shift_dim=32, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shift_dim = shift_dim
        self.lap = modelLap.LapLoss()
        self.feature_ofnet = Loaded_Modified_IFNet(pretrained_model=pretrained_model)
        self.unet_0to1 = Unet(hidden_dim=self.hidden_dim, shift_dim=self.shift_dim)
        self.decoder =  nn.Sequential()

        
    def forward(self, x, forwardContext_d2, forwardContext_d4, backwardContext_d2, backwardContext_d4):
        
        gt = x[:, 6:] # In inference time, gt is None
        #  x_rev = torch.cat([img1,img0,gt], dim=1)
        forward_shiftedFeature, backward_shiftedFeature  = self.feature_ofnet(x) # 32*h*w
        
        featureUnet = self.unet_0to1(forward_shiftedFeature, backward_shiftedFeature, forwardContext_d2, forwardContext_d4, backwardContext_d2, backwardContext_d4)
        
        predictimage = self.decoder(featureUnet)
        
        # presavemerge = [0,0,0]
        # predictimage = torch.clamp(merged[2] + tmp, 0, 1)

        # Temporally put timeframeFeatrues here
        # modified to tanh as output ~[-1,1]
        # tPredict = tmp[:, :3] * 2 - 1
        # predictimage = tmp
        #merged[2] = torch.clamp(tPredict, 0, 1)
        
        loss_ssimd = 0
        loss_pred = (self.lap(predictimage,gt)).mean() 
        loss_mse = (predictimage - gt)**2
        loss_mse = loss_mse.mean()
        # loss_pred = self.lpips_model(predictimage, gt).mean()

        #loss_pred = (((merged[2] - gt) **2).mean(1,True)**0.5).mean()
        # loss_tea = 0
        merged_teacher = predictimage *0  # not used. just to avoid error
        flow_teacher= predictimage *0     # not used. just to avoid error
        mask_list = [flow_teacher, flow_teacher,flow_teacher]
        flow_list = [flow_teacher,flow_teacher,flow_teacher]
        return flow_list, merged_teacher, predictimage, flow_teacher, merged_teacher, loss_ssimd, loss_mse, loss_pred
            #  flow,      mask,           merged,       flow_teacher, merged_teacher, loss_ssimd, loss_tea, loss_pred, loss_ssimd = self.feature_ofnet(imgs_and_gt, fallfeatures[i], ballfeatures[-(1+i)])
    
        # return flow_list, mask_list[2], merged[2], flow_teacher, merged_teacher, loss_ssimd, loss_tea, loss_pred
            

class VSRbackbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.feature_ofnet = newMergeIFnet(shift_dim=32, pretrained_model=pretrained)
        
        self.convgru = ConvGRUFeatures(hidden_dim=32)


    def forward(self, allframes):
        # allframes 0<1>2<3>4<5>6
        # IFnet module
        #b, n, c, h, w = allframes_N.shape()
        #allframes = allframes_N.view(b,n*c,h,w)
        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_ssimd = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_mse = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_ssimd = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')

        output_allframes = []
        output_onlyteacher = []
        flow_list = []
        flow_teacher_list = []
        mask_list = []
        fallfeatures_2d, fallfeatures_4d, ballfeatures_2d, ballfeatures_4d = self.convgru(allframes)
        for i in range(0, 3, 1):
            img0 = allframes[:, 6*i:6*i+3]
            gt = allframes[:, 6*i+3:6*i+6]
            img1 = allframes[:, 6*i+6:6*i+9]
            imgs_and_gt = torch.cat([img0,img1,gt],dim=1)
            flow, mask, merged, flow_teacher, merged_teacher,loss_ssimd,  loss_mse, loss_pred = self.feature_ofnet(imgs_and_gt, fallfeatures_2d[i], fallfeatures_4d[i], ballfeatures_2d[-(i+1)], ballfeatures_4d[-(1+i)])
            # flow, mask, merged, flow_teacher, merged_teacher, loss_ssimd = self.flownet(allframes)
            Sum_loss_ssimd += loss_ssimd 
            Sum_loss_context += loss_pred
            Sum_loss_mse +=loss_mse
            Sum_loss_ssimd  += loss_ssimd
            output_allframes.append(img0)
            # output_allframes.append(merged[2])
            output_allframes.append(merged)
            flow_list.append(flow)
            flow_teacher_list.append(flow_teacher)
            output_onlyteacher.append(merged_teacher)
            mask_list.append(mask)

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()
        
        img6 = allframes[:,-3:] 
        output_allframes.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        pass


        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_onlyteacher, Sum_loss_ssimd, Sum_loss_context, Sum_loss_mse
