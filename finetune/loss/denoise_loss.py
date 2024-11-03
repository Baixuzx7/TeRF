import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import imageio.v3 as imageio
import torchvision
import os

""" Ref Zero Noise to Noise  """


class denoise_loss(nn.Module):
    def __init__(self):
        super(denoise_loss, self).__init__() 
        self.measure = nn.MSELoss()

    def pair_downsampler(self, img): 
        c = img.shape[1]
        
        filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
        filter1 = filter1.repeat(c,1, 1, 1)
        
        filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
        filter2 = filter2.repeat(c,1, 1, 1)
        
        output1 = F.conv2d(img, filter1, stride=2, groups=c)
        output2 = F.conv2d(img, filter2, stride=2, groups=c)
        
        return output1, output2

