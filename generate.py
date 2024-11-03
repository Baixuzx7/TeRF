import os
import cv2
import shutil
import random
import numpy as np
from PIL import Image
import imageio.v3 as imageio
import regex as re

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"

import torch
import torchvision

""" Large Model """
from union import UnionLLM,UnionLVM,UnionFusion
from finetune.trainers import FineTuneTrainer

""" Tools """
from tools import querries_pre_processing
from tools import masks_pre_processing
from tools import task_processing_pipline
from tools import tf2np,tf2img,img_crop_merge
from tools import display_cuda_info
from tools import recorder_merge
from tools import setup_seed,setup_device


def generate_text_prompts(Unionlvm,image,N):
    tags,tag_list = Unionlvm.inference_tags(image) # tags : str  tag_list : list
    object_set = random.sample(tag_list, N)
    effect_list = ["increase the visibility of", "decrease the visibility",
        "highlight the saliency of",
        "reduce the noise of",
        "improve the visibility of", 
        "blur the", 
        "lower the illumination of",
        "enhance the",
        "denoise the",
        "Improve the clarity of" ,
        "Reduce the brightness of",
        "Enhance the details of",
        "Remove the graininess of",
        "Brighten the",
        "Obscure the identity of",
        "Increase the focus on",
        "Refine the texture of",
        "Eliminate the background noise from",
        "Make the background appear out of",
        "Reduce the intensity of", 
        "Enhance the texture of",
        "Intensify the lighting of",
        "Dim the",
        "Emphasize the prominence of",
        "Amplify the visibility of",
        "Apply a Gaussian blur to" ,
        "Remove the noise from",
        "Increase the intensity of"
        "Blur the edges of" ,
        "enhance the saliency of"]
    effect_set = random.sample(effect_list, N)
    querries = ""
    for i in range(N):
        querries = querries + effect_set[i] + ' ' + object_set[i] + ', '
    return querries[:-2]

def main():
    image_vi_filePath = "./data/vision/MSRS/train/vi/"
    image_ir_filePath = "./data/vision/MSRS/train/ir/"
    
    image_vi_listdir = os.listdir(image_vi_filePath)
    image_vi_listdir.sort(key=lambda x: int(x[:-5]))
    image_ir_listdir = os.listdir(image_ir_filePath)
    image_ir_listdir.sort(key=lambda x: int(x[:-5]))
    image_number = len(image_vi_listdir)

    setup_device('lvm')
    Vision_LM = UnionLVM(device = 'cuda')    
    Fusion_LM = UnionFusion(device = 'cuda') 
    Finetune_LM = FineTuneTrainer(device = 'cuda')
    setup_device('llm')
    Text_LM = UnionLLM()           

    for image_id in range(image_number):
        image_vi_path = image_vi_filePath + image_vi_listdir[image_id]
        image_ir_path = image_ir_filePath + image_ir_listdir[image_id]
        print('Processing Image Name ',image_vi_listdir[image_id])

        setup_device('lvm')
        image_vi = Image.open(image_vi_path) 
        querries  = generate_text_prompts(Unionlvm = Vision_LM,image = image_vi,N = 4)
        print(querries)
        
        setup_device('llm')
        text_prompts  = querries
        object_list,task_list,outputs_dict,outputs_name = querries_pre_processing(Unionllm = Text_LM, querries = text_prompts)
        print(object_list,task_list) 

        setup_device('lvm')
        image_vi = Image.open(image_vi_path)
        image_ir = Image.open(image_ir_path)
        image_fn = Fusion_LM.process(image_vi,image_ir)
        image_finetune = image_fn
        display_cuda_info()
        
        all_masks = Vision_LM.generate_all_masks(image_vi)
        image_recorder = [np.asarray(image_vi),np.asarray(image_fn)]
        mask_recorder = [np.concatenate([np.asarray(image_ir)[:,:,np.newaxis],np.asarray(image_ir)[:,:,np.newaxis],np.asarray(image_ir)[:,:,np.newaxis]],axis=-1),np.asarray(all_masks)]

        task_num = len(object_list)
        for procedure_id in range(task_num):
            object_description = object_list[procedure_id]
            task_description = task_list[procedure_id]
            print('Process Analysis --->  Task {}, {}, Segemantation Region ---> Mask {}'.format(task_description,outputs_dict[procedure_id]['effect'],object_description))
            masks_union = masks_pre_processing(Unionlvm = Vision_LM,raw_image = image_vi,text_prompt = object_description, procedure_id = procedure_id)
            print('Process --->  Finetune')
            image_finetune = task_processing_pipline(Finetune_LM,image_ir,image_vi,image_finetune,masks_union,task_description)
            print('Process --->  Save')
            mask_recorder.append(Vision_LM.generate_color_masks(masks_union))
            image_recorder.append(image_finetune)
            print('Finished')

        
        torch.cuda.empty_cache()
        image_loger = recorder_merge(image_recorder,mask_recorder)
        imageio.imwrite('./outputs/pipelines/{}_{}.jpg'.format(image_id,re.sub(r'[^A-Za-z0-9 ]+', '',str(outputs_name))),image_loger.astype(np.uint8))

if __name__ == '__main__':
    print('Hello World')