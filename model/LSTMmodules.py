import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.laplacian import *
from model.warplayer import warp
from model.refine import *
from model.myContext import *
from model.loss import *
from model.myLossset import *
from model.Attenions import *
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
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    

class FBwardExtractor(nn.Module):
    # input 3 outoput 6
    def __init__(self, in_plane=3, out_plane=c):
        super().__init__()
        # set stride = 2 to downsample
        # as for 224*224, the current output shape is 112*112
        self.out_plane = out_plane
        self.fromimage = conv(in_plane, c, kernel_size=3, stride=1, padding=1)
        self.downsample = conv(c, 2*c, kernel_size=3, stride=2, padding=1)

        self.conv0 = nn.Sequential(
            conv(2*c, 2*c, 3, 1, 1),
            conv(2*c, 4*c, 3, 1, 1),
            conv(4*c, 2*c, 3, 1, 1),
            conv(2*c, self.out_plane, 3, 1, 1),
            )
        self.forwardFeatureList = []
        self.upsample = nn.ConvTranspose2d(in_channels=self.out_plane, out_channels=self.out_plane, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, allframes):
        # all frames: B*21*H*W  --> 
        # x is concated frames [0,2,4,6] -> [(4*3),112,112] 
        forwardFeatureList = []
        for i in range(0,4):
            x = allframes[:, 3*i:3*i+3].clone()
            y = self.fromimage(x)  # 224*224 -> 224*224
            
            x = self.downsample(y)  # 224*224 -> 112*112
            x = self.conv0(x)       # Pass through conv layers
            
            # Upsample and add to the original y tensor
            x = self.upsample(x) + y
            
            forwardFeatureList.append(x)
            # self.forwardFeatureList.append(x)


        return forwardFeatureList
    # Output: BNCHW



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


class Modified_IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lap = LapLoss()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)
        self.block_tea = IFBlock(16+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.lpips_model = lpips.LPIPS(net='vgg').to(device)

    def forward(self, x, forwardContext, backwardContext, scale=[4,2,1]):
        # forwardContext/backwardContext is forwardFeature[i], only pick up the one for the current interpolation
        # final_merged, loss = self.mIFnetframesset[i], for(wardFeature[3*i], backwardFeature[3*i+2])
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        stdv = np.random.uniform(0.0, 5.0)
        img0 = (img0 + stdv * torch.randn(*img0.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp
        img1 = (img1 + stdv * torch.randn(*img1.shape).cuda()).clamp(0.0, 255.0)  # Add noise and clamp       

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        # eps = 1e-8

        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        # Temporally put timeframeFeatrues here
        # modified to tanh as output ~[-1,1]
        tmp = self.unet(img0, img1, warped_img0, warped_img1, forwardContext, backwardContext,mask, flow, c0, c1)
        # tPredict = tmp[:, :3] * 2 - 1
        # predictimage = tmp
        #merged[2] = torch.clamp(tPredict, 0, 1)
        predictimage = torch.clamp(merged[2] + tmp, 0, 1)
        loss_pred += SSIMD(predictimage, gt)
        #loss_pred = (((merged[2] - gt) **2).mean(1,True)**0.5).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()

        return flow_list, mask_list[2], predictimage, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred
     
# c = 48
class ConvGRUFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        # current encoder: all frames ==> all features 
        self.img2Fencoder = FBwardExtractor()
        self.img2Bencoder = FBwardExtractor()
        self.forwardgru = unitConvGRU()
        self.backwardgru = unitConvGRU()
        self.hidden_dim = 128
        self.Fusion = Modified_IFNet()

    def forward(self, allframes):
        # aframes = allframes_N.view(b,n*c,h,w)
        # Output: BNCHW
        fcontextlist = []
        bcontextlist = []
        fallfeatures = self.img2Fencoder(allframes)
        ballfeatures = self.img2Bencoder(allframes)
        b, _, h, w = allframes.size()


        # forward GRU 
        # h' = gru(h,x)
        # Method A: zero initialize Hiddenlayer
        forward_hidden_initial = torch.zeros((b, self.hidden_dim, h, w),device=device )
        backward_hidden_initial = torch.zeros((b, self.hidden_dim, h, w), device=device)
        # n=4
        # I skipped the 0 -> first image
        for i in range(0,4):
            if i == 0:
                fhidden = self.forwardgru(forward_hidden_initial, fallfeatures[i])
                bhidden = self.backwardgru(backward_hidden_initial, ballfeatures[-i-1])
            else:
                fhidden = self.forwardgru(fhidden, fallfeatures[i])
                bhidden = self.backwardgru(bhidden, ballfeatures[-i-1])
                fcontextlist.append(fhidden)
                bcontextlist.append(bhidden)

        return fcontextlist, bcontextlist
        # return forwardFeature, backwardFeature
        # Now iterate through septuplet and get three inter frames





class VSRbackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.Fusion = Modified_IFNet()
        self.convgru = ConvGRUFeatures()


    def forward(self, allframes):
        # allframes 0<1>2<3>4<5>6
        # IFnet module
        #b, n, c, h, w = allframes_N.shape()
        #allframes = allframes_N.view(b,n*c,h,w)
        Sum_loss_context = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_distill = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        Sum_loss_tea = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'gpu')
        output_allframes = []
        output_onlyteacher = []
        flow_list = []
        flow_teacher_list = []
        mask_list = []
        fallfeatures, ballfeatures = self.convgru(allframes)
        for i in range(0, 3, 1):
            img0 = allframes[:, 6*i:6*i+3]
            gt = allframes[:, 6*i+3:6*i+6]
            img1 = allframes[:, 6*i+6:6*i+9]
            imgs_and_gt = torch.cat([img0,img1,gt],dim=1)
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill, loss_tea, loss_pred = self.Fusion(imgs_and_gt, fallfeatures[i], ballfeatures[-(1+i)])
            # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(allframes)
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

            # The way RIFE compute prediction loss and 
            # loss_l1 = (self.lap(merged[2], gt)).mean()
            # loss_tea = (self.lap(merged_teacher, gt)).mean()
        
        img6 = allframes[:,-3:] 
        output_allframes.append(img6)
        output_allframes_tensors = torch.stack(output_allframes, dim=1)
        pass


        return flow_list, mask_list, output_allframes_tensors, flow_teacher_list, output_onlyteacher, Sum_loss_distill, Sum_loss_context, Sum_loss_tea

# unused
class unitGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.outputs = []
        pass
    def forward(self, inputs):
      pass
    """  
        for X in inputs:
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z )
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r )
            H_tilda = torch.tanh(
                (X @ W_xh) + ((R * H) @ W_hh + b_h)
            )
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            self.outputs.append(Y)

        return torch.cat(outputs, dim=0),
    """

# Unused
class scaledImageFeature(nn.Module):
    def __init__(self, image_H=224, image_W=224):
        self.shape = [image_H, image_W]


    
    def forward(self, allframes, ):
        # allframes: 0 2 4 6
        # feature = 
        pass

# Unused
class GRUcontext(nn.Module):
    def __init__(self):
        
        self.bGRU = nn.GRU()
        self.fGRU = nn.GRU()

    def forward(self, allfeatures):
        pass

# unused
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros(
        (batch_size, num_hiddens), device=device
    ))

