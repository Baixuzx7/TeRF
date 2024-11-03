
import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio.v3 as imageio
import torchvision

from torch.utils.tensorboard import SummaryWriter

""" Load the necessary loss functions """
from finetune.networks import base_fusion_model
from finetune.networks import fineture_fusion_model
from finetune.networks import fineture_lumin_model
from finetune.networks import fineture_denoise_model
from finetune.networks import fineture_blur_model


from finetune.loss.biasIR_loss import fusion_bias_IR_loss
from finetune.loss.biasVIS_loss import fusion_bias_VIS_loss
from finetune.loss.lumin_loss import lumin_loss
from finetune.loss.fusion_loss import fusion_loss
from finetune.loss.denoise_loss import denoise_loss
 
class FineTuneTrainer(nn.Module):
    def __init__(self,device):
        super(FineTuneTrainer,self).__init__()
        self.device = device
        self.initialize(0)

    def initialize(self,task_id):
        self.finetune_model = self.select_pretrained_model(task_id)
        self.finetune_model.init_weight()
        self.select_optimizer(task_id)
        self.finetune_model.to(self.device)

        self.writer = SummaryWriter('./blog')
    
        self.criterion = self.select_loss_function(task_id)


    def forward(self,task_id,image_ir,image_vi,image_fn):
    
        return self.cal_loss_fn(task_id,image_ir.to(self.device),image_vi.to(self.device),image_fn.to(self.device))
    

    def cal_loss_fn(self,task_id,image_ir,image_vi,image_fn = None):
        
        if task_id == 0:    
            loss = self.criterion(image_ir.to(self.device),image_vi.to(self.device),image_fn.to(self.device))
            return loss
        elif task_id == 1:  
            noisy1, noisy2 = self.criterion.pair_downsampler(image_fn.to(self.device))
            pred1 =  noisy1 - self.finetune_model(noisy1.to(self.device))
            pred2 =  noisy2 - self.finetune_model(noisy2.to(self.device))
            loss_res = 1/2*(F.mse_loss(noisy1.to(self.device),pred2.to(self.device))+F.mse_loss(noisy2.to(self.device),pred1.to(self.device)))
            noisy_denoised =  image_fn.to(self.device) - self.finetune_model(image_fn.to(self.device))
            denoised1, denoised2 = self.criterion.pair_downsampler(noisy_denoised.to(self.device))
            loss_cons=1/2*(F.mse_loss(pred1.to(self.device),denoised1.to(self.device)) + F.mse_loss(pred2.to(self.device),denoised2.to(self.device)))
            loss = loss_res + loss_cons
            return loss
        elif task_id == 2: 
            illumination, reflectance, noise = self.finetune_model(image_fn.to(self.device))
            loss = self.criterion(image_fn, illumination, reflectance, noise)
            return loss
        elif task_id == 3: 
            return 0
        elif task_id == 4: 
            image_ft = self.finetune_model(image_ir.to(self.device),image_vi.to(self.device),image_fn.to(self.device)) 
            loss = self.criterion(image_ir.to(self.device),image_vi.to(self.device),image_ft.to(self.device))
            return loss
        elif task_id == 5: 
            image_ft = self.finetune_model(image_ir.to(self.device),image_vi.to(self.device),image_fn.to(self.device)) 
            loss = self.criterion(image_ir.to(self.device),image_vi.to(self.device),image_ft.to(self.device))
            return loss
        elif task_id == 6: 
            illumination, reflectance, noise = self.finetune_model(image_fn.to(self.device))
            loss = self.criterion(image_fn, illumination, reflectance, noise)
            return loss
        else:
            return 0


    def select_optimizer(self,task_id):
        if task_id == 0:   
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=1e-3) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.1)
        elif task_id == 1:  
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=2e-5) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif task_id == 2:  
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=5e-4)     
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif task_id == 3:  
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=2e-5) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif task_id == 4:  
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=8e-5) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif task_id == 5:  
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=8e-5) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif task_id == 6: 
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=5e-4) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        else:
            self.optimizer = torch.optim.Adam(self.finetune_model.parameters(),lr=2e-5) 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        
    def select_pretrained_model(self,task_id):
        if task_id == 0:    
            return base_fusion_model(device=self.device,pretrained_path = "/data/BaiXuYa/MutilModal/Re/TextControlFusionMS/fusion/checkpoint/net/Best.pth").to(self.device)
        elif task_id == 1: 
            return fineture_denoise_model()
        elif task_id == 2:  
            return fineture_lumin_model()    
        elif task_id == 3:
            return fineture_blur_model()
        elif task_id == 4:  
            return base_fusion_model(device=self.device,pretrained_path = "/data/BaiXuYa/MutilModal/Re/TextControlFusionMS/fusion/checkpoint/net/Best.pth").to(self.device)
        elif task_id == 5: 
            return base_fusion_model(device=self.device,pretrained_path = "/data/BaiXuYa/MutilModal/Re/TextControlFusionMS/fusion/checkpoint/net/Best.pth").to(self.device)
        elif task_id == 6: 
            return fineture_lumin_model()
        else:
            return base_fusion_model(device=self.device,pretrained_path = "/data/BaiXuYa/MutilModal/Re/TextControlFusionMS/fusion/checkpoint/net/Best.pth").to(self.device)
        

    def select_loss_function(self,task_id):
        if   task_id == 0:  
            return fusion_loss()    
        elif task_id == 1:  
            return denoise_loss()
        elif task_id == 2:  
            return lumin_loss(self.device)
        elif task_id == 3: 
            return None
        elif task_id == 4:  
            return fusion_bias_VIS_loss()
        elif task_id == 5:  
            return fusion_bias_IR_loss()
        elif task_id == 6:  
            return lumin_loss(self.device)        
        return None
    
    

if __name__ == '__main__':
    print('Hello Wolrd')