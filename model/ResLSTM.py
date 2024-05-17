import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.refine import *
from model.myContext import *  
from model.myLossset import *  
from model.laplacian import *
# Attention test
# import model.Attenions as att
from model.loss import *
import model.Resnet as resnet
from model.myLossset import *
import model.Attenions as att
import yaml


class single_conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, stride=1,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_planes,out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = nn.BatchNorm2d(out_planes)
        self.LeReLU = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.LeReLU(x)
        return x
    
class Extractor_conv(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cnn1 = single_conv(3,64,3,1,1)
        self.cnn2 = single_conv(64,128)
        self.cnn3 = single_conv(128,128)
        self.cnn4 = single_conv(128,128)
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        return x


   
c = 64
class unitConvGRU(nn.Module):
    # Formula:
    # I_t = Sigmoid(Conv(x_t;W_{xi}) + Conv(h_{t-1};W_{hi}) + b_i)
    # F_t = Sigmoid(Conv(x_t;W_{xf}) + Conv(h_{t-1};W_{hi}) + b_i)
    def __init__(self, hidden_dim=128, input_dim=c):
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


class unitConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.conv_f = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_c = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_o = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, hidden:tuple):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)  # concatenate along the channel dimension
        
        f_t = torch.sigmoid(self.conv_f(combined))  # forget gate
        i_t = torch.sigmoid(self.conv_i(combined))  # input gate 
        o_t = torch.sigmoid(self.conv_o(combined))  # output gate
        c_tilde = torch.tanh(self.conv_c(combined))  # candidate cell state
        
        c_t = f_t * c_prev + i_t * c_tilde  # cell state update
        h_t = o_t * torch.tanh(c_t)  # hidden state update

        return h_t, (h_t, c_t)

class fusion_Fimage_hidden(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.bn = nn.BatchNorm2d(num_features=out_planes)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.fussConv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1)
        )
    def forward(self, hidden_state, image_feature):
        fuss = torch.concat([hidden_state, image_feature],dim=1)
        fuss = self.fussConv(fuss)
        fuss = self.bn(fuss)
        fuss = self.leakyRelu(fuss)
        return fuss
        

class ResLSTMflowUnit(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.image_extractor = Extractor_conv()
        self.convlstm = unitConvLSTM(hidden_dim=self.hidden_size, input_dim=128)
        self.fusion = fusion_Fimage_hidden(in_planes=128+self.hidden_size, out_planes=128)
        self.attention = att.CBAM(in_channel=128)
        self.Res1 = resnet.BasicResNetUnit(in_channels=128, out_channels=128)
        self.Res2 = resnet.NoReluBasicResNetUnit(in_channels=128, out_channels=128)
        self.finalact = nn.LeakyReLU(inplace=True)
        
        
    def forward(self, RRinput, selected_frame, pre_hidden_cell):
        '''
        Unit of ResLSTM. Stack it to finish the backbone.

        Args:
        - selected_frame (torch.Tensor): Tensor of shape (b, n, c, h, w) representing a batch of frame sequences.0<1>2<3>4<5>6
            a) b, n, c, h, w = allframes_N.shape()
        - selected_frame (torch.Tensor): Tensor of shape (b, 1, c, h, w) representing the currently selected frame. 
            1. b, 1, c, h, w = selected_frame
            2. forward:0 -> 2 -> 4
            3. backward:6 -> 4 -> 2
        - hidden_state: Hidden state and cell state for the ConvLSTM.

        '''

        # X_1 = RRinput
        X_2 = self.Res1(RRinput)
        # h_t, (h_t, c_t)
        hidden, curr_hidden_cell= self.convlstm(X_2, pre_hidden_cell)
        
        freature_image = self.image_extractor(selected_frame)
        fusion_feature = self.fusion(freature_image , hidden)
        X_3= self.attention(fusion_feature)
        X_3_r = self.Res2(X_3)
        X_4 = self.finalact(X_3_r + fusion_feature)
        output = X_4   
            
        return curr_hidden_cell, output, X_4

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
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

class Update_Contextnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2(3, 16)
        self.conv2 = Conv2(16, 2*16)
        self.conv3 = Conv2(2*16, 4*16)
        self.conv4 = Conv2(4*16, 8*16)
    
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


class Update_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = Conv2(17, 2*16)
        self.down1 = Conv2(4*16, 4*16)
        self.down2 = Conv2(8*16, 8*16)
        self.down3 = Conv2(16*16, 16*16)
        self.up0 = deconv(32*16, 8*16)
        self.up1 = deconv(16*16, 4*16)
        self.up2 = deconv(8*16, 2*16)
        self.up3 = deconv(4*16, 16)
        self.conv = nn.Conv2d(16, 3, 3, 1, 1)

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
       
       
class Loaded_Modified_IFNet(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.lap = LapLoss()
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
        self.contextnet = Contextnet()
        self.unet = Unet()
        
        # self.contextnet = pretrained_model.contextnet()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def forward(self, x, forwardContext, backwardContext, scale_list=[4,2,1]):
        # forwardContext/backwardContext is forwardFeature[i], only pick up the one for the current interpolation
        # final_merged, loss = self.mIFnetframesset[i], for(wardFeature[3*i], backwardFeature[3*i+2])
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        # stdv = np.random.uniform(0.0, 5.0)
        # img0 = (img0 + stdv * torch.randn(*img0.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp
        # img1 = (img1 + stdv * torch.randn(*img1.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp       

        loss_distill = 0
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
            merged.append((warped_img0, warped_img1))
# ----------------


        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, forwardContext, backwardContext,mask, flow, c0, c1)
        presavemerge = [0,0,0]
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            presavemerge[i] = merged[i]
            # if gt.shape[1] == 3:
                # loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                # loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        predictimage = torch.clamp(merged[2] + tmp, 0, 1)

        # Temporally put timeframeFeatrues here
        # modified to tanh as output ~[-1,1]
        # tPredict = tmp[:, :3] * 2 - 1
        # predictimage = tmp
        #merged[2] = torch.clamp(tPredict, 0, 1)
        
        loss_ssimd = SSIMD(predictimage, gt)*0.5
        loss_pred = (self.lap(predictimage,gt)).mean() + loss_ssimd
        
        # loss_pred = self.lpips_model(predictimage, gt).mean()

        #loss_pred = (((merged[2] - gt) **2).mean(1,True)**0.5).mean()
        loss_tea = 0
        merged_teacher = merged[2]
        flow_teacher= flow

        return flow_list, mask_list[2], predictimage, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred
        # return flow_list, mask_list[2], merged[2], flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred
            

class IFNet_update(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7+4, c=90)
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)
        self.block_tea = IFBlock(10+4, c=90)
        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, x, scale_list=[4, 2, 1]):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        loss_cons = 0
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
            merged.append((warped_img0, warped_img1))
        '''
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, 1:4] * 2 - 1
        '''
        for i in range(3):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            # merged[i] = torch.clamp(merged[i] + res, 0, 1)        
        return flow_list, mask_list[2], merged

         
class VSRbackbone(nn.Module):
    def __init__(self,pretrainedIFnet):
        super().__init__()
        
        self.hidden_size = 128
        self.initializer = Extractor_conv()
        self.forwardLSTM = ResLSTMflowUnit(hidden_size=self.hidden_size)
        self.backwardLSTM = ResLSTMflowUnit(hidden_size=self.hidden_size)
        # Download and load the origin RIFE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the state dictionary
        self.IFnet = Loaded_Modified_IFNet(pretrainedIFnet)  


    def forward(self, allframes):
        # allframes 0<1>2<3>4<5>6
        # IFnet module
        
        b, c, h, w = allframes.shape
        reshaped_frames = allframes.view(b, 7, 3, h, w)
        h0 = torch.zeros(b,self.hidden_size,h,w).to(allframes.device)
        c0 = torch.zeros(b,self.hidden_size,h,w).to(allframes.device)
        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_distill = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        output_allframes = []
        output_onlyteacher = []
        flow_list = []
        flow_teacher_list = []
        mask_list = []
        forward_output_list = []
        backward_output_list = []
        initial_hc = (h0, c0)
        for m in range(3):
            i = 2*m
            if m == 0:
                # assume allframes:b,n,c,h,w
                f_RRinput = self.initializer(reshaped_frames[:,0])
                b_RRinput = self.initializer(reshaped_frames[:,-1])
                forward_curr_hidden_cell, forward_output, f_X_4 = self.forwardLSTM(RRinput=f_RRinput, selected_frame=reshaped_frames[:,i+2], pre_hidden_cell=initial_hc)
                backward_curr_hidden_cell, backward_output, b_X_4 = self.backwardLSTM(RRinput=b_RRinput, selected_frame=reshaped_frames[:,-i-3], pre_hidden_cell=initial_hc)
            else:
                forward_curr_hidden_cell, forward_output, f_X_4 = self.forwardLSTM(RRinput=f_X_4, selected_frame=reshaped_frames[:,i+2], pre_hidden_cell=forward_curr_hidden_cell)
                backward_curr_hidden_cell, backward_output, b_X_4 = self.backwardLSTM(RRinput=b_X_4, selected_frame=reshaped_frames[:,-i-3], pre_hidden_cell=backward_curr_hidden_cell)
            forward_output_list.append(forward_output)
            backward_output_list.append(backward_output)
        for i in range(0, 3, 1):
            img0 = allframes[:, 6*i:6*i+3]
            gt = allframes[:, 6*i+3:6*i+6]
            img1 = allframes[:, 6*i+6:6*i+9]
            imgs_and_gt = torch.cat([img0,img1,gt],dim=1)
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred = self.IFnet(imgs_and_gt, forward_output_list[i], backward_output_list[-(1+i)])
            Sum_loss_distill += loss_distill 
            Sum_loss_context += 1/3 * loss_pred
            Sum_loss_tea +=loss_tea
            output_allframes.append(img0)
            # output_allframes.append(merged[2])
            output_allframes.append(merged)
            flow_list.append(flow)
            flow_teacher_list.append(flow_teacher)
            output_onlyteacher.append(merged_teacher)
            mask_list.append(mask)
        
        img6 = allframes[:,-3:] 
        output_allframes.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_onlyteacher, Sum_loss_distill, Sum_loss_context, Sum_loss_tea

        