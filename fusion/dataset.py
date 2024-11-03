import torch
import os
import imageio.v3 as imageio
import torchvision
import cv2
import numpy as np
import torchvision.transforms.functional as ttf

class VIR_NIR_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, istrain):
        self.root = root
        
        self.transform_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])
        
        self.random_crop = istrain

        self.images_IR = list(sorted(os.listdir(os.path.join(self.root, 'ir'))))
        self.images_IR.sort(key=lambda x: int(x[:-4]))
        self.images_VI = list(sorted(os.listdir(os.path.join(self.root, 'vi'))))
        self.images_VI.sort(key=lambda x: int(x[:-4]))
        

    def __getitem__(self, item):
        image_IR_path = os.path.join(self.root, 'ir', self.images_IR[item])
        image_VI_path = os.path.join(self.root, 'vi', self.images_VI[item])
        
        
        image_IR_L = imageio.imread(image_IR_path)     # GrayScale

        image_VI_RGB = imageio.imread(image_VI_path)   # RGB  

        image_VI_YCrCb = cv2.cvtColor(image_VI_RGB, cv2.COLOR_RGB2YCR_CB) # YCbCr

        if self.transform_to_tensor is not None:
            image_IR_L = self.transform_to_tensor(image_IR_L)
            image_VI_RGB = self.transform_to_tensor(image_VI_RGB)
            image_VI_YCrCb = self.transform_to_tensor(image_VI_YCrCb)

        if self.random_crop:
            # randomly crops
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(image_IR_L, output_size=(150,150))
            image_IR_L = ttf.crop(image_IR_L, i, j, h, w)
            image_VI_RGB = ttf.crop(image_VI_RGB, i, j, h, w)
            image_VI_YCrCb = ttf.crop(image_VI_YCrCb, i, j, h, w)
        
        image_VI_Y = image_VI_YCrCb[0,:,:].unsqueeze(0)

        return image_IR_L, image_VI_Y, image_VI_RGB, image_VI_YCrCb
        # PIL.image PIL.image PytorchTensor PytorchTensor

    def __len__(self):
        return len(self.images_IR)
     

if __name__ == '__main__': 
    datasets = VIR_NIR_Dataset('../data/vision/MSRS/train')
    image_ir,image_vi_Y,image_vi_RGB,image_VI_YCrCb = datasets[0]
    train_loader = torch.utils.data.DataLoader(dataset=datasets, batch_size=16, shuffle=True)
    for i,(image_ir,image_vi_Y,image_vi_RGB,image_VI_YCrCb) in enumerate(train_loader):
        print(image_ir.size())
        exit()

    print(image_vi_Y.shape,image_vi_Y.max(),image_vi_Y.min())

    # print(image_vi_YCrCb.shape,image_vi_YCrCb.max(),image_vi_YCrCb.min())
    # (image_ir,image_vi_Y,image_vi_RGB,image_VI_YCrCb)