import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.model import RDN as FusionNetwork


class fineture_fusion_network(nn.Module):
    def __init__(self) -> None:
        super(fineture_fusion_network,self).__init__()
        self.fusion_finetune = FusionNetwork()
        

    def forward(self, image_ir,image_vi):
        outputs = self.fusion_finetune(image_ir,image_vi)
        return outputs

class base_fusion_model(nn.Module):
    def __init__(self,device,pretrained_path = None) -> None:
        super(base_fusion_model,self).__init__()
        self.device = device
        self.weight_path = pretrained_path
        self.fusion_finetune = FusionNetwork().to(device)
  
    def forward(self, image_ir,image_vi,image_fn = None):
        outputs = self.fusion_finetune(image_ir.to(self.device),image_vi.to(self.device))
        return outputs
    
    def init_weight(self,conv_type = 'trunc', bias_type = None ,bn_type = None): 
        if self.weight_path == None: 
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if conv_type == 'kaiming':
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, val=0.0)
                    elif conv_type == 'trunc':
                        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, val=0)
                    else:
                        nn.init.constant_(m.weight, val=1/9)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, val=0.0)
                    pass
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                pass
        else: 
            self.fusion_finetune.load_state_dict(torch.load(self.weight_path,map_location=self.device))
        pass

class fineture_fusion_model(nn.Module):
    def __init__(self,device,pretrained_path = None) -> None:
        super(fineture_fusion_model,self).__init__()
        self.device = device
        self.weight_path = pretrained_path 

        self.fusion_finetune = fineture_fusion_network().to(self.device)
        

    def forward(self, image_ir,image_vi,image_fn = None): 
        outputs = self.fusion_finetune(image_ir,image_vi)
        return outputs
    
    def init_weight(self,conv_type = None, bias_type = None ,bn_type = None): 
        if self.weight_path == None: 
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if conv_type == 'kaiming':
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, val=0.0)
                    elif conv_type == 'trunc':
                        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, val=0)
                    else:
                        nn.init.constant_(m.weight, val=1/12)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, val=0.0)
                    pass
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                pass
        else: 
            self.fusion_finetune.load_state_dict(torch.load(self.weight_path,map_location=self.device))
        pass

class fineture_lumin_model(nn.Module):
    def __init__(self):
        super(fineture_lumin_model, self).__init__()

        self.illumination_net = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),nn.ReLU(),nn.Conv2d(32, 64, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),nn.ReLU(),nn.Conv2d(32, 1, 3, 1, 1))

        self.reflectance_net = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),nn.ReLU(),nn.Conv2d(32, 64, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),nn.ReLU(),nn.Conv2d(32, 3, 3, 1, 1))

        self.noise_net = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),nn.ReLU(),nn.Conv2d(32, 64, 3, 1, 1),nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),nn.ReLU(),nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, input):
        illumination = torch.sigmoid(self.illumination_net(input))
        reflectance = torch.sigmoid(self.reflectance_net(input))
        noise = torch.tanh(self.noise_net(input))

        return illumination, reflectance, noise
    
    def init_weight(self,conv_type = 'trunc', bias_type = None ,bn_type = None): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if conv_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                elif conv_type == 'trunc':
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0)
                else:
                    nn.init.constant_(m.weight, val=1/9)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                pass


class fineture_denoise_model(nn.Module):
    def __init__(self,n_chan=3,chan_embed=48):
        super(fineture_denoise_model, self).__init__()
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self.conv4 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv6 = nn.Conv2d(chan_embed, n_chan, 1)


    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.conv6(x)
        return x
    
    def init_weight(self,conv_type = 'trunc', bias_type = None ,bn_type = None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if conv_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                elif conv_type == 'trunc':
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0)
                else:
                    nn.init.constant_(m.weight, val=1/9)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                pass


class fineture_denoise_model(nn.Module):
    def __init__(self,n_chan=3,chan_embed=48):
        super(fineture_denoise_model, self).__init__()
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self.conv4 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv6 = nn.Conv2d(chan_embed, n_chan, 1)


    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.conv6(x)
        return x
    
    def init_weight(self,conv_type = 'trunc', bias_type = None ,bn_type = None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if conv_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                elif conv_type == 'trunc':
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.05)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0)
                else:
                    nn.init.constant_(m.weight, val=1/9)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, val=0.0)
                pass


class fineture_blur_model(nn.Module):
    def __init__(self,n_chan=1,blur_kernel=7):
        super(fineture_blur_model, self).__init__()
        self.blur_kernel = blur_kernel
        self.conv1 = nn.Conv2d(n_chan,n_chan,blur_kernel,padding=blur_kernel//2)        
        self.act = nn.ReLU() 

    def forward(self, x, t):
        for i in range(t):
            x = self.act(self.conv1(x))
        return x
    
    def init_weight(self,conv_type = None , bias_type = None ,bn_type = None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, val=1/self.blur_kernel/self.blur_kernel)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)


if __name__ == '__main__':
    print('Hello World')
