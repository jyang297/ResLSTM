import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
# from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
# from model.IFNet_BIVSR_7images import *
from model.LSTMwithAttention import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
class Model:
    def __init__(self, pretrainedModel, local_rank=-1, arbitrary=False):

        self.flownet = VSRbackbone(pretrainedModel)
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        # if local_rank != -1:
        #    self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()
        
    def parameter_loaded_train(self):
        # Load parameters for Ori_IFnet

    
        self.device()
        self.flownet.train()
    
    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
        if rank == 0:
            checkpoint = torch.load(f'{path}/flownet.pkl', map_location=self.device)
            #self.flownet.load_state_dict(convert(checkpoint))
            #model_path = 'intrain_log/flownet.pkl'
            # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            # Load the state dictionary
            self.flownet.load_state_dict(checkpoint)
            self.flownet.to(self.device)
        
    def save_model(self, path, rank=0):
        if not os.path.exists(path):
            os.makedirs(path)
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, allframes, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        if training:
            self.train()
        else:
            self.eval()
        flow_list, mask_list, output_allframes, flow_teacher_list, output_onlyteacher, Sum_loss_distill, Sum_loss_context, Sum_loss_tea,Sum_loss_ssimd = self.flownet(allframes)
        # flow_list, mask_list, output_allframes, flow_teacher_list, output_onlyteacher, Sum_loss_distill, Sum_loss_context, Sum_loss_tea
        

        # loss_l1 = (self.lap(merged[2], gt)).mean()
        # loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            torch.autograd.set_detect_anomaly(True)
            self.optimG.zero_grad()
            loss_G = Sum_loss_context + Sum_loss_tea + Sum_loss_distill* 0.01 + Sum_loss_ssimd*0.5# when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet.parameters(), max_norm=1.0)
            self.optimG.step()
        else:
            pass
            # flow_teacher = flow[2]
        return output_allframes, {
            'merged_tea': output_onlyteacher[-1],
            'mask': mask_list[-1],
            #'mask_tea': mask,
            'flow': flow_list[-1],
            'flow_tea': flow_teacher_list[-1],
            'Sum_loss_context': Sum_loss_context,
            'Sum_loss_tea': Sum_loss_tea,
            'Sum_loss_distill': Sum_loss_distill,
            'Sum_loss_ssimd':Sum_loss_ssimd
            }