import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
from einops import rearrange

class sub_pixel(nn.Module):
    def __init__(self, scale):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
    
class downsample(nn.Module):
    def __init__(self,in_feat,scale):
        super(downsample, self).__init__()
        self.conv_down = nn.Conv2d(in_feat,in_feat,kernel_size=scale,stride=scale,padding=0)
        self.conv_ext = nn.Conv2d(in_feat,in_feat,kernel_size=3,stride=1,padding=1)
        self.conv_inn = nn.Conv2d(in_feat,in_feat,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        y = self.conv_inn(self.conv_ext(self.conv_down(x)))
        return y
        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.activation = nn.PReLU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        # x = F.gelu(x)
        x = self.activation(x)
        x = self.project_out(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, input_channel=4, output_channel=3, nDenselayer=6, nFeat=64, scale=2, growthRate=32, num_refinement_blocks=2):
        super(RDN, self).__init__()
        # Feature Extraction
        self.FEconv = nn.Sequential(
            nn.Conv2d(input_channel, nFeat, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
        )
        # Residual Dense Blocks x 3 
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)

        # Global Feature Fusion (GFF)
        self.GFEconv = nn.Sequential(
            nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True),
            nn.PReLU(),
        )
         

        self.refinement = nn.Sequential(*[TransformerBlock(dim=nFeat, num_heads=1, ffn_expansion_factor=2.66,
                             bias=False, LayerNorm_type='WithBias') for i in range(num_refinement_blocks)])

        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        self.conv_down = downsample(nFeat,scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, output_channel, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        F_0 = self.FEconv(x)  
        F_1 = self.RDB1(F_0) 
        F_2 = self.RDB2(F_1) 
        F_3 = self.RDB3(F_2) 
        FF = torch.cat((F_1, F_2, F_3), 1)
 
        t = self.GFEconv(FF)
        FDF = self.refinement(t) +  F_0

        us = self.conv_up(FDF)
        us = self.upsample(us)
        us = self.conv_down(us)
        output = self.conv3(us)
        return output # + 0.5 * (x1 + x2)
    

from torch.utils.tensorboard import SummaryWriter
import cv2 
import os
from torch.optim import lr_scheduler
from fusion.loss import fusion_loss
import imageio
import numpy as np


class FusionTrainer(nn.Module):
    def __init__(self,fusion_model,opt = None):
        super(FusionTrainer,self).__init__()
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() else 'cpu')
        self.fusion_model = fusion_model.to(self.device)

        self.optimizer = torch.optim.Adam(self.fusion_model.parameters(),lr=opt.lr,betas=(opt.beta1,opt.beta2))        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_scheduler_step, gamma=opt.lr_scheduler_decay)
        self.writer = SummaryWriter(opt.writer_dir)
        self.criterion = fusion_loss().to(self.device)

        if opt.iscontinue:
            print('Continue Training --> trainer Loads Parameters : {}.pth'.format(opt.continue_load_name))
            self.load_model_parameters(self.fusion_model,opt.net_params_dir,name = opt.continue_load_name)
            self.load_optim_parameters(self.optimizer,opt.opt_params_dir,name = opt.continue_load_name)
        else:
            # print('Error the parameters of fusion model are not initialized')
            print('Parameters of fusion model are initialized')
            # self.fusion_model.initialize()
            # exit(0)
 
    def forward(self,image_ir,image_vi):
        image_fusion = self.fusion_model(image_ir.to(self.device),image_vi.to(self.device))
        loss = self.criterion(image_ir.to(self.device),image_vi.to(self.device),image_fusion.to(self.device)) 
        return loss
    
    def save_model_parameters(self,model,save_path,name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(),os.path.join(save_path,'{}.pth'.format(name)))

    def load_model_parameters(self,model,load_path,name):
        if not os.path.exists(load_path):
            exit('No such file, Please Check the path : ', load_path,name,'.pth')
        else:
            print('Loading Model',os.path.join(load_path,'{}.pth'.format(name)))
        model.load_state_dict(torch.load(os.path.join(load_path,'{}.pth'.format(name)),self.device))

    def save_optim_parameters(self,optimizer,save_path,name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(optimizer.state_dict(),os.path.join(save_path,'{}.pth'.format(name)))

    def load_optim_parameters(self,optimizer,load_path,name):
        if not os.path.exists(load_path):
            exit('No such file, Please Check the path : ', load_path,name,'.pth')
        else:
            print('Loading Optim',os.path.join(load_path,'{}.pth'.format(name)))
        optimizer.load_state_dict(torch.load(os.path.join(load_path,'{}.pth'.format(name)),self.device))

    def np2tf(self,image_np):
        if len(image_np.shape) == 2:
            image_tf = torch.from_numpy(image_np).contiguous().unsqueeze(0).unsqueeze(0)
        else:
            image_tf = torch.from_numpy(image_np).contiguous().permute(2,0,1).unsqueeze(0)
        
        return image_tf

    def tf2np(self,image_tf):
        n,c,h,w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        
        return image_np
    
    def tf2img(self,image_tf):
        image_np = self.tf2np(torch.clamp(image_tf,min=0.,max=1.))
        image_np = (image_np * 255).astype(np.uint8)
        return image_np
    
    def image_merge(self,image_Y_tf,image_YCrCb_tf): 
        with torch.no_grad():
            image_merge_tf = image_YCrCb_tf
            image_merge_tf[:,0,:,:] = image_Y_tf
            image_YCrCb_np = (self.tf2np(torch.clamp(image_merge_tf,min=0.,max=1.)) * 255).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_YCrCb_np, cv2.COLOR_YCR_CB2RGB).astype(np.uint8)
        return image_rgb

    def save_image(self,image,save_path,name,type = 'jpg'):
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        imageio.imwrite(os.path.join(save_path,'{}.{}'.format(name,type)),image.astype(np.uint8))



if __name__ == "__main__":
    print('Hello World')
    device = 'cuda'
    image_vi = torch.rand([1,3,322,478])
    image_ir = torch.rand([1,1,322,478])

    net = RDN().to(device)
    t = net(image_vi.to(device),image_ir.to(device))
    print(t.shape)

    print('Finished')




