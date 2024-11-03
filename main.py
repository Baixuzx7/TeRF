import os
import cv2
import shutil
import random
import numpy as np
from PIL import Image
import imageio.v3 as imageio

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

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
from tools import analysis_metrics


def main():
    image_vi_filePath = "./data/vision/RoadScene/vi/"
    image_ir_filePath = "./data/vision/RoadScene/ir/"
    image_vi_listdir = os.listdir(image_vi_filePath)
    image_vi_listdir.sort(key=lambda x: int(x[:-4]))
    image_ir_listdir = os.listdir(image_ir_filePath)
    image_ir_listdir.sort(key=lambda x: int(x[:-4]))
    image_number = len(image_vi_listdir)
    image_id = 0
    image_vi_path = image_vi_filePath + image_vi_listdir[image_id]
    image_ir_path = image_ir_filePath + image_ir_listdir[image_id]
    print('Processing Image Name ',image_vi_listdir[image_id])
    
    setup_device('llm')
    Text_LM = UnionLLM()          
    
    setup_device('lvm')
    Vision_LM = UnionLVM(device = 'cuda')    
    Fusion_LM = UnionFusion(device = 'cuda') 
    Finetune_LM = FineTuneTrainer(device = 'cuda')
    
    image_vi = Image.open(image_vi_path)
    image_ir = Image.open(image_ir_path)
    image_fn = Fusion_LM.process(image_vi,image_ir)

    image_fn.save('./outputs/fuses/{}.jpg'.format(image_id))

    setup_device('llm')
    text_prompts  = "make car more salient, increase the brightness of trees"
    object_list,task_list,outputs_dict,outputs_name = querries_pre_processing(Unionllm = Text_LM, querries = text_prompts)
    print(object_list,task_list) 

    setup_device('lvm')
    image_finetune = image_fn 
    all_masks = Vision_LM.generate_all_masks(image_vi)
    image_recorder = [np.asarray(image_vi),np.asarray(image_fn)]
    mask_recorder = [np.concatenate([np.asarray(image_ir)[:,:,np.newaxis],np.asarray(image_ir)[:,:,np.newaxis],np.asarray(image_ir)[:,:,np.newaxis]],axis=-1),np.asarray(all_masks)]

    task_num = len(object_list)
    for procedure_id in range(task_num):
        object_description = object_list[procedure_id]
        task_description = task_list[procedure_id]
        print('Process Analysis --->  Task {}, {}, Segemantation Region ---> Mask {}'.format(task_description,outputs_dict[procedure_id]['effect'],object_description))
        setup_device('lvm')
        print('Process --->  Finetuning')
        masks_union = masks_pre_processing(Unionlvm = Vision_LM,raw_image = image_fn,text_prompt = object_description, procedure_id = procedure_id,start_id=0,end_id=None)
        image_finetune = task_processing_pipline(Finetune_LM,image_ir,image_vi,image_finetune,masks_union,task_description)
        print('Process --->  Save') 
        mask_recorder.append(Vision_LM.generate_color_masks(masks_union)) 
        image_recorder.append(image_finetune) 
        
    print('Finished')

    torch.cuda.empty_cache()
    image_loger = recorder_merge(image_recorder,mask_recorder) 
    imageio.imwrite('./variants.jpg',image_loger.astype(np.uint8))

    print('Before (metric value of {})'.format(image_vi_listdir[image_id]))
    analysis_metrics(image_vi,image_ir,image_fn,save_flag = False)
    print('After  (metric value of {})'.format(image_vi_listdir[image_id]))
    analysis_metrics(image_vi,image_ir,image_finetune,save_flag = True)


if __name__ == '__main__':
    setup_seed(3407) 
    if os.path.exists('./outputs/masks'):
        shutil.rmtree('./outputs/masks')
    main()