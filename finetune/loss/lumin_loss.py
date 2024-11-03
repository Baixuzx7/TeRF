import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import cv2

"""     illuminance Loss     """
 
class lumin_loss(nn.Module):
    def __init__(self,device):
        super(lumin_loss, self).__init__()
        self.device = device
        self.illu_factor = 1
        self.reflect_factor = 1
        self.noise_factor = 5000
        self.reffac = 1
        self.g_kernel_size = 5
        self.g_padding = 2
        self.sigma = 3
        self.kx = cv2.getGaussianKernel(self.g_kernel_size,self.sigma)
        self.ky = cv2.getGaussianKernel(self.g_kernel_size,self.sigma)
        self.gaussian_kernel = np.multiply(self.kx,np.transpose(self.ky))
        self.gaussian_kernel = torch.FloatTensor(self.gaussian_kernel).unsqueeze(0).unsqueeze(0).to(self.device)


    def reconstruction_loss(self, image, illumination, reflectance, noise):
        reconstructed_image = illumination*reflectance+noise
        return torch.norm(image-reconstructed_image, 1)


    def gradient(self, img):
        height = img.size(2)
        width = img.size(3)
        gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
        gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
        gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
        gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
        gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
        gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
        gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
        gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
        return gradient_h*gradient2_h, gradient_w*gradient2_w


    def normalize01(self, img):
        minv = img.min()
        maxv = img.max()
        return (img-minv)/(maxv-minv)


    def gaussianblur3(self, input):
        slice1 = F.conv2d(input[:,0,:,:].unsqueeze(1), weight=self.gaussian_kernel, padding=self.g_padding)
        slice2 = F.conv2d(input[:,1,:,:].unsqueeze(1), weight=self.gaussian_kernel, padding=self.g_padding)
        slice3 = F.conv2d(input[:,2,:,:].unsqueeze(1), weight=self.gaussian_kernel, padding=self.g_padding)
        x = torch.cat([slice1,slice2, slice3], dim=1)
        return x


    def illumination_smooth_loss(self, image, illumination):
        gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
        max_rgb, _ = torch.max(image, 1)
        max_rgb = max_rgb.unsqueeze(1)
        gradient_gray_h, gradient_gray_w = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_illu_h, gradient_illu_w = self.gradient(illumination)
        weight_h = 1/(F.conv2d(gradient_gray_h, weight=self.gaussian_kernel, padding=self.g_padding)+0.0001)
        weight_w = 1/(F.conv2d(gradient_gray_w, weight=self.gaussian_kernel, padding=self.g_padding)+0.0001)
        weight_h.detach()
        weight_w.detach()
        loss_h = weight_h * gradient_illu_h
        loss_w = weight_w * gradient_illu_w
        max_rgb.detach()
        return loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1)


    def reflectance_smooth_loss(self, image, illumination, reflectance):
        gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
        gradient_gray_h, gradient_gray_w = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_reflect_h, gradient_reflect_w = self.gradient(reflectance)
        weight = 1/(illumination*gradient_gray_h*gradient_gray_w+0.0001)
        weight = self.normalize01(weight)
        weight.detach()
        loss_h = weight * gradient_reflect_h
        loss_w = weight * gradient_reflect_w
        refrence_reflect = image/illumination
        refrence_reflect.detach()
        return loss_h.sum() + loss_w.sum() + self.reffac*torch.norm(refrence_reflect - reflectance, 1)


    def noise_loss(self, image, illumination, reflectance, noise):
        weight_illu = illumination
        weight_illu.detach()
        loss = weight_illu*noise
        return torch.norm(loss, 2)
    
    def forward(self, img_tensor, illumination, reflectance, noise):
    
        loss_recons = self.reconstruction_loss(img_tensor, illumination, reflectance, noise)
        loss_illu = self.illumination_smooth_loss(img_tensor, illumination)
        loss_reflect = self.reflectance_smooth_loss(img_tensor, illumination, reflectance)
        loss_noise = self.noise_loss(img_tensor, illumination, reflectance, noise)
        loss = loss_recons + self.illu_factor*loss_illu + self.reflect_factor*loss_reflect + self.noise_factor*loss_noise
        return loss
    

if __name__ == '__main__':
    print('Hello World')
