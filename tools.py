import torch
import imageio.v3 as imageio
import torchvision
import cv2 
import numpy as np
import random
import math

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sklearn.metrics as skm
from skimage.metrics import structural_similarity as compare_ssim
from scipy.fftpack import dctn
from scipy.signal import convolve2d
from scipy.ndimage import sobel, generic_gradient_magnitude


from tqdm import tqdm
import os

def tf2np(image_tf):
    n,c,h,w = image_tf.size()
    assert n == 1
    if c == 1:
        image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
    else:
        image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    
    return image_np

def tf2img(image_tf):
    image_np = tf2np(torch.clamp(image_tf,min=0.,max=1.))
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def img_crop_merge(source,target,mask):
    assert len(source.shape) == len(target.shape)
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.uint8)
    if len(source.shape) > 2:
        h,w,c = source.shape
        multi_channel_mask = (mask[:,:,np.newaxis].repeat(c,axis=-1) // 255).astype(np.float32)
        image_target = target * multi_channel_mask
        background = source * (1-multi_channel_mask)
        image_merge = image_target + background
    else:
        image_target = target * (mask // 255).astype(np.float32)
        background = source * (1-(mask // 255)).astype(np.float32)
        image_merge = image_target + background 
    return image_merge.astype(np.uint8)


def mask_to_bbox_point(mask):
    _,mask_bw = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    assert len(np.unique(mask)) <= 2
    if len(np.unique(mask)) == 2:
        h_set,w_set = np.where(mask_bw == 255)
        left_bound,right_bound = min(w_set),max(w_set)
        up_bound,down_bound = min(h_set),max(h_set)
    else:
        left_bound,right_bound = 0,mask_bw.shape[1]
        up_bound,down_bound = 0,mask_bw.shape[0]
    return left_bound,right_bound,up_bound,down_bound


def mask_to_bbox(image_src,left_bound,right_bound,up_bound,down_bound):
    if len(image_src.shape) == 2:
        image_dst = image_src[up_bound:down_bound+1,left_bound:right_bound+1] 
    else:
        image_dst = image_src[up_bound:down_bound+1,left_bound:right_bound+1,:]
    return image_dst


def crop_to_entity(image_crop,mask,left_bound,right_bound,up_bound,down_bound):
    h,w = mask.shape
    c = image_crop.shape[-1]
    image_tmp = np.zeros([h,w,c])
    image_tmp[up_bound:down_bound+1,left_bound:right_bound+1,:] = image_crop
    image_outputs = image_tmp * (mask[:,:,np.newaxis].repeat(3,axis=-1) // 255)
    return image_outputs

def display_cuda_info():
    info = ''
    available_gpus_counts = torch.cuda.device_count()
    for id in range(available_gpus_counts):    
        p = torch.cuda.get_device_properties(id)     
        allocated_gpus_memory = torch.cuda.memory_allocated(id)
        maximum_gpus_memory = torch.cuda.max_memory_allocated(id)
        allocated_memory_percent = allocated_gpus_memory / p.total_memory * 100
        info += f'CUDA:{id} ({p.name}, {allocated_gpus_memory / (1 << 20):5.0f}/{p.total_memory / (1 << 20):5.0f}MiB), {allocated_memory_percent:2.2f}%\n'
        pass
    print(info)

def recorder_merge(image_set,mask_set,sort_way = 1):
    assert len(image_set) == len(mask_set)
    image_sample = np.concatenate([image_set[0],image_set[1]],axis=int(sort_way))
    mask_sample = np.concatenate([mask_set[0],mask_set[1]],axis=int(sort_way))
    h,w,c = image_sample.shape
    N = len(image_set)
    for idx in range(2,N):
        image_sample = np.concatenate([image_sample,image_set[idx]],axis=int(sort_way)) 
        mask_sample = np.concatenate([mask_sample,mask_set[idx]],axis=int(sort_way)) 
    image_output = np.concatenate([image_sample,mask_sample],axis=int(1-sort_way))
    return image_output


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_device(keys):
    if keys == 'llm':
        torch.set_default_dtype(torch.float32)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            print('Default Device for LLM: ', device)
    else:
        torch.set_default_dtype(torch.float32)
        device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            print('Default Device for LVM: ', device)

def querries_pre_processing(Unionllm,querries):
    outputs = Unionllm.process(querries)
    outputs_dict = Unionllm.convert_dict(outputs)
    object_list,task_list = Unionllm.convert_task_object(outputs_dict)
    return object_list,task_list,outputs_dict,outputs


def masks_pre_processing(Unionlvm,raw_image,text_prompt = None, procedure_id = None, start_id = 0, end_id = None):
    if text_prompt == 'unknown':
        t = np.asarray(raw_image)
        h,w = t.shape[0],t.shape[1]
        return np.zeros([h,w]).astype(np.uint8)
    else:
        masks = Unionlvm.process(raw_image,text_prompt)
        masks_set,logit_set = masks[0],masks[1]
        n,c,h,w = masks_set.size()
        print('logit_set: ',logit_set)
        masks_savage = np.zeros([h,w], dtype=bool)
        if logit_set != []: 
            if start_id == None:
                start_id = 0
            if end_id == None:
                end_id = n
            for i in range(start_id,end_id): 
                mask_np = (masks_set[i,:,:,:].squeeze(0)).cpu().numpy()
                if not os.path.exists('./outputs/masks/procedure_{}'.format(int(procedure_id))):
                    os.makedirs('./outputs/masks/procedure_{}'.format(int(procedure_id)))
                imageio.imwrite('./outputs/masks/procedure_{}/{}_{}.jpg'.format(procedure_id,text_prompt,logit_set[i]),(mask_np * 255).astype(np.uint8))
                masks_savage = masks_savage + mask_np
        else:
            masks_savage = np.zeros([h,w], dtype=bool)
        return (masks_savage * 255).astype(np.uint8)
    
def task_process_denoise(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 1):
    image_ir_np,image_vi_np,image_fn_np = np.asarray(image_ir),np.asarray(image_vi),np.asarray(image_fn)

    dilate_kernel = np.ones((5, 5), dtype=np.uint8)
    image_mask = cv2.dilate(mask, dilate_kernel,5)

    image_vi_np_target = image_vi_np 
    image_ir_np_target = image_ir_np 
    image_vi_target = torchvision.transforms.ToTensor()(image_vi_np_target).unsqueeze(0)
    image_ir_target = torchvision.transforms.ToTensor()(image_ir_np_target).unsqueeze(0)
    image_fn_pt = torchvision.transforms.ToTensor()(image_fn_np).unsqueeze(0)

    finetune_trainer.initialize(task_id)   
    max_epochs, loss_minimun = 1000, 999  
    epoch_bar = tqdm(range(max_epochs))
    for epoch in epoch_bar:
        """ Train """
        finetune_trainer.train()
        finetune_trainer.optimizer.zero_grad()
        loss = finetune_trainer(task_id = 1,image_ir = image_ir_target,image_vi = image_vi_target,image_fn = image_fn_pt)
        loss.backward()
        finetune_trainer.optimizer.step()
        epoch_bar.set_description_str('Epoch:{}/{}  Loss: {:.6f} '.format(epoch, max_epochs ,loss.item()))
        finetune_trainer.scheduler.step()
        pass
    with torch.no_grad():
        pred_pt = image_fn_pt.to(finetune_trainer.device) - finetune_trainer.finetune_model(image_fn_pt.to(finetune_trainer.device))
        enhanced_image = tf2img(pred_pt)
    image_finetune = img_crop_merge(image_fn_np,enhanced_image,mask) 
    return image_finetune.astype(np.uint8)


def task_process_lowlightEnhance(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 2):
    image_ir_np,image_vi_np,image_fn_np = np.asarray(image_ir),np.asarray(image_vi),np.asarray(image_fn)
    image_vi_np_target = image_vi_np 
    image_ir_np_target = image_ir_np 
    image_vi_target = torchvision.transforms.ToTensor()(image_vi_np_target).unsqueeze(0)
    image_ir_target = torchvision.transforms.ToTensor()(image_ir_np_target).unsqueeze(0)
    image_fn_pt = torchvision.transforms.ToTensor()(image_fn_np).unsqueeze(0)
    
    finetune_trainer.initialize(task_id)   
    Numerator = math.log10(1/2)
    average_lumin = 0
    for i in range(3):
        average_lumin = average_lumin + np.mean(image_vi_np[:,:,i].astype(np.float32))/255
    denominator = math.log10(average_lumin / 3.0)
    gamma_vals = Numerator / denominator
    if gamma_vals == 1:
        gamma_vals = gamma_vals - random.random()
    finetune_trainer.criterion.gamma = min(Numerator / denominator,0.2)  # Hyperparameters that need to be set
    print('finetune_trainer.criterion.gamma: ',finetune_trainer.criterion.gamma)
    max_epochs, loss_minimun = 1000, 999  
    epoch_bar = tqdm(range(max_epochs))
    for epoch in epoch_bar:
        finetune_trainer.train()
        finetune_trainer.optimizer.zero_grad()
        loss = finetune_trainer(task_id,image_ir = image_ir_target,image_vi = image_vi_target,image_fn = image_fn_pt)
        loss.backward()
        finetune_trainer.optimizer.step()
        epoch_bar.set_description_str('Epoch:{}/{}  Loss: {:.6f} '.format(epoch, max_epochs ,loss.item()))
        finetune_trainer.scheduler.step()
        pass
    with torch.no_grad():
        illumination, reflectance, noise = finetune_trainer.finetune_model(image_fn_pt.to(finetune_trainer.device))
        adjust_illu = torch.pow(illumination.to(finetune_trainer.device), finetune_trainer.criterion.gamma)
        res_image = adjust_illu.to(finetune_trainer.device)*((image_fn_pt.to(finetune_trainer.device)-noise.to(finetune_trainer.device))/illumination.to(finetune_trainer.device))
        enhanced_image = tf2img(res_image)
    image_finetune = enhanced_image
    return image_finetune.astype(np.uint8)

def task_process_blur(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 3):
    image_ir_np,image_vi_np,image_fn_np = np.asarray(image_ir),np.asarray(image_vi),np.asarray(image_fn)
    dilate_kernel = np.ones((5, 5), dtype=np.uint8)
    image_mask = cv2.dilate(mask, dilate_kernel,5)
    image_fn_pt = torchvision.transforms.ToTensor()(image_fn_np).unsqueeze(0)

    finetune_trainer.initialize(task_id)  
    with torch.no_grad():
        res_image = torch.zeros_like(image_fn_pt)
        for channel in range(image_fn_pt.size(1)):
            res_image[:,channel:channel+1,:,:] = finetune_trainer.finetune_model(image_fn_pt[:,channel,:,:].unsqueeze(1).to(finetune_trainer.device),t=3)
        enhanced_image = tf2img(res_image)
    image_finetune = img_crop_merge(image_fn_np,enhanced_image,mask)     
    return image_finetune.astype(np.uint8)


def task_process_visible(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 4):
    image_ir_np,image_vi_np,image_fn_np = np.asarray(image_ir),np.asarray(image_vi),np.asarray(image_fn)
    image_vi_np_target = image_vi_np 
    image_ir_np_target = image_ir_np 
    image_fn_np_target = image_fn_np

    image_vi_target = torchvision.transforms.ToTensor()(image_vi_np_target).unsqueeze(0)
    image_ir_target = torchvision.transforms.ToTensor()(image_ir_np_target).unsqueeze(0)
    image_fn_pt = torchvision.transforms.ToTensor()(image_fn_np_target).unsqueeze(0)
     
    finetune_trainer.initialize(task_id) 
    max_epochs = 70
    epoch_bar = tqdm(range(max_epochs))
    for epoch in epoch_bar:
        finetune_trainer.train()
        finetune_trainer.optimizer.zero_grad()
        loss = finetune_trainer(task_id,image_ir = image_ir_target,image_vi = image_vi_target,image_fn = image_fn_pt)
        loss.backward()
        finetune_trainer.optimizer.step()
        epoch_bar.set_description_str('Epoch:{}/{}  Loss: {:.6f} '.format(epoch, max_epochs ,loss.item()))
        finetune_trainer.scheduler.step()
    with torch.no_grad():
        image_ft_taget_pt = finetune_trainer.finetune_model(image_ir_target.to(finetune_trainer.device),image_vi_target.to(finetune_trainer.device),image_fn_pt.to(finetune_trainer.device))
        image_ft_taget_np = (tf2np(torch.clamp(image_ft_taget_pt,min=0.,max=1.)) * 255).astype(np.uint8)
    image_finetune = image_ft_taget_np 
    return image_finetune.astype(np.uint8)

def task_process_infrared(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 5):
    image_ir_np,image_vi_np,image_fn_np = np.asarray(image_ir),np.asarray(image_vi),np.asarray(image_fn)

    image_vi_np_target = image_vi_np 
    image_ir_np_target = image_ir_np 
    image_fn_np_target = image_fn_np

    image_vi_target = torchvision.transforms.ToTensor()(image_vi_np_target).unsqueeze(0)
    image_ir_target = torchvision.transforms.ToTensor()(image_ir_np_target).unsqueeze(0)
    image_fn_pt = torchvision.transforms.ToTensor()(image_fn_np_target).unsqueeze(0)
    
    finetune_trainer.initialize(task_id)
    max_epochs = 100
    epoch_bar = tqdm(range(max_epochs))
    for epoch in epoch_bar:
        finetune_trainer.train()
        finetune_trainer.optimizer.zero_grad()
        loss = finetune_trainer(task_id,image_ir = image_ir_target,image_vi = image_vi_target,image_fn = image_fn_pt)
        loss.backward()
        finetune_trainer.optimizer.step()
        epoch_bar.set_description_str('Epoch:{}/{}  Loss: {:.6f} '.format(epoch, max_epochs ,loss.item()))
        finetune_trainer.scheduler.step()
    with torch.no_grad():
        image_ft_taget_pt = finetune_trainer.finetune_model(image_ir_target.to(finetune_trainer.device),image_vi_target.to(finetune_trainer.device),image_fn_pt.to(finetune_trainer.device))
        image_ft_taget_np = (tf2np(torch.clamp(image_ft_taget_pt,min=0.,max=1.)) * 255).astype(np.uint8)
    image_finetune = image_ft_taget_np 
    return image_finetune.astype(np.uint8)


def task_process_highlightcorrect(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 6):
    image_ir_np,image_vi_np,image_fn_np = np.asarray(image_ir),np.asarray(image_vi),np.asarray(image_fn)

    image_vi_np_target = image_vi_np
    image_ir_np_target = image_ir_np 
    image_vi_target = torchvision.transforms.ToTensor()(image_vi_np_target).unsqueeze(0)
    image_ir_target = torchvision.transforms.ToTensor()(image_ir_np_target).unsqueeze(0)
    image_fn_pt = torchvision.transforms.ToTensor()(image_fn_np).unsqueeze(0)

    finetune_trainer.initialize(task_id)
    Numerator = math.log10(1/2.0)
    average_lumin = 0
    for i in range(3):
        average_lumin = average_lumin + np.mean(image_vi_np[:,:,i].astype(np.float32)) / 255
    denominator = math.log10(average_lumin / 3.0)
    gamma_vals = Numerator / denominator
    if gamma_vals == 1:
        gamma_vals = gamma_vals + random.random()
    finetune_trainer.criterion.gamma = max(Numerator / denominator,2.5) # Hyperparameters that need to be set
    print('finetune_trainer.criterion.gamma: ',finetune_trainer.criterion.gamma)
    max_epochs, loss_minimun = 500, 999  
    epoch_bar = tqdm(range(max_epochs))
    for epoch in epoch_bar:
        finetune_trainer.train()
        finetune_trainer.optimizer.zero_grad()
        loss = finetune_trainer(task_id,image_ir = image_ir_target,image_vi = image_vi_target,image_fn = image_fn_pt)
        loss.backward()
        finetune_trainer.optimizer.step()
        epoch_bar.set_description_str('Epoch:{}/{}  Loss: {:.6f} '.format(epoch, max_epochs ,loss.item()))
        finetune_trainer.scheduler.step()
        pass
    with torch.no_grad():
        illumination, reflectance, noise = finetune_trainer.finetune_model(image_fn_pt.to(finetune_trainer.device))
        adjust_illu = torch.pow(illumination.to(finetune_trainer.device), finetune_trainer.criterion.gamma)
        res_image = adjust_illu.to(finetune_trainer.device)*((image_fn_pt.to(finetune_trainer.device)-noise.to(finetune_trainer.device))/illumination.to(finetune_trainer.device))
        enhanced_image = tf2img(res_image)
    image_finetune = enhanced_image
    return image_finetune.astype(np.uint8)


def task_processing_pipline(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id):
    if task_id == 0:    
        print('Process ---> Initialize ---> FusionNet') 
        return image_fn
    elif task_id == 1:
        print('Process ---> Initialize ---> DenoiseNet')
        return task_process_denoise(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 1)
    elif task_id == 2:  
        print('Process ---> Initialize ---> LuminNet')
        return task_process_lowlightEnhance(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 2)
    elif task_id == 3:
        print('Process ---> Initialize ---> BlurNet')
        return task_process_blur(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 3)
    elif task_id == 4: 
        print('Process ---> Initialize ---> FusionNet')
        return task_process_visible(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 4)
    elif task_id == 5:  
        print('Process ---> Initialize ---> FusionNet')
        return task_process_infrared(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 5)
    elif task_id == 6: 
        print('Process ---> Initialize ---> LuminNet')
        return task_process_highlightcorrect(finetune_trainer,image_ir,image_vi,image_fn,mask,task_id = 6)
    else:
        return image_fn

import pandas as pd

def analysis_metrics(image_vi,image_ir,image_fn,save_flag = False):
    image_vi,image_ir,image_fn = np.asarray(image_vi),np.asarray(image_ir),np.asarray(image_fn)
    AG_val = analysis_AG(image_fn)
    CC_val = analysis_CC(image_vi,image_ir,image_fn)
    EN_val = analysis_EN(image_fn)
    PSNR_val = analysis_PSNR(image_vi,image_ir,image_fn)
    SSIM_val = analysis_ssim(image_vi,image_ir,image_fn)
    VIF_val = analysis_VIF(image_vi,image_ir,image_fn)
    SCD_val = analysis_SCD(image_vi,image_ir,image_fn)
    SF_val = analysis_SF(image_fn)
    SD_val = analysis_SD(image_fn)
    MI_val = analysis_MI(image_vi,image_ir,image_fn)

    print('Evaluate mean ---> AG :  {:.4f} CC :  {:.4f} EN :  {:.4f} PSNR :  {:.4f} SSIM :  {:.4f} VIF :  {:.4f} SCD :  {:.4f} SF :  {:.4f} SD :  {:.4f} MI :  {:.4f}   '.format(
    AG_val,CC_val,EN_val,PSNR_val,SSIM_val,VIF_val,SCD_val,SF_val,SD_val,MI_val))

    if save_flag:
        data_save = np.asarray([
            AG_val,CC_val,EN_val,PSNR_val,SSIM_val,VIF_val,SCD_val,SF_val,SD_val,MI_val
        ])
        pd.set_option("display.precision", 4)
        data_save = data_save[np.newaxis,:]
        data_df=pd.DataFrame(data_save) 
        writer = pd.ExcelWriter('./outputs/metrics/metric.xlsx') 
        data_df.to_excel(writer,'page_1') 
        writer._save()


def analysis_MI(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    B = image_ir.astype(np.uint8)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    A = np.reshape(A, -1)
    B = np.reshape(B, -1)
    F = np.reshape(F, -1)
    
    haf = skm.mutual_info_score(A, F)
    hbf = skm.mutual_info_score(B, F)
    return haf + hbf

def analysis_ssim(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.uint8)
    B = image_ir.astype(np.uint8)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.uint8)

    ssim_AF = compare_ssim(A,F)
    ssim_BF = compare_ssim(B,F)

    ssim_mean = (ssim_AF + ssim_BF) * 0.5
    return ssim_mean


def analysis_AG(image):
    img = image.astype(np.float32)
    if len(image.shape) == 2:
        img = img[:,:,np.newaxis]
    h,w,c = img.shape
    g = np.zeros(c)
    for i in range(c):
        image_channel = img[:,:,i]
        [dy, dx] = np.gradient(image_channel)
        s = np.sqrt((np.power(dx,2) + np.power(dy,2))/2); 
        g[i] = np.sum(s) / h / w; 
    val = np.mean(g)
    return val

def analysis_CC(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC

def analysis_VIF(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF


def analysis_SCD(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    print(corr2(F - B, A),corr2(F - A, B))
    r = corr2(F - B, A) + corr2(F - A, B)
    return r


def analysis_PSNR(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A)**2))/(m*n)
    MSE_BF = np.sum(np.sum((F - B)**2))/(m*n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255/np.sqrt(MSE))
    return PSNR

def analysis_MSE(image_vi,image_ir,image_fn):
    A = cv2.cvtColor(image_vi,cv2.COLOR_RGB2GRAY).astype(np.float32)
    B = image_ir.astype(np.float32)
    F = cv2.cvtColor(image_fn,cv2.COLOR_RGB2GRAY).astype(np.float32)
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A)**2))/(m*n)
    MSE_BF = np.sum(np.sum((F - B)**2))/(m*n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE


def analysis_SD(image_array):
    image_array = image_array.astype(np.float32)
    if len(image_array.shape) == 2:
        image_array = image_array[:,:,np.newaxis]
    m, n, c = image_array.shape
    SD = np.zeros(c)
    for i in range(c):
        u = np.mean(image_array[:,:,i])
        SD[i] = np.sqrt(np.sum(np.sum((image_array[:,:,i] - u) ** 2)) / (m * n))
    return SD.mean()


def analysis_EN(image_array):
    if len(image_array.shape) == 2:
        image_array = image_array[:,:,np.newaxis]
    h,w,c = image_array.shape
    entropy = 0
    for i in range(c):
        histogram, bins = np.histogram(image_array[:,:,i].astype(np.uint8), bins=256, range=(0, 255))
        histogram = histogram / float(np.sum(histogram))
        entropy = entropy - np.sum(histogram * np.log2(histogram + 1e-7))
    entropy = entropy / c
    return entropy

def analysis_SF(image):
    image_array = np.array(image)
    RF = np.diff(image_array, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF ** 2)))
    CF = np.diff(image_array, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF ** 2)))
    SF = np.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(4-scale+1)+1
        win = fspecial_gaussian((N, N), N/5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = convolve2d(ref*ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist*dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref*dist, win, mode='valid') - mu1_mu2

        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12

        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        num += np.sum(np.log10(1+g**2 * sigma1_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1+sigma1_sq/sigma_nsq))
    vifp = num/den
    return vifp




def analysis_FMI(ima, imb, imf, feature, w):
    ima = np.double(ima)
    imb = np.double(imb)
    imf = np.double(imf)

    if feature == 'none': 
        aFeature = ima
        bFeature = imb
        fFeature = imf
    elif feature == 'gradient':  
        aFeature = generic_gradient_magnitude(ima, sobel)
        bFeature = generic_gradient_magnitude(imb, sobel)
        fFeature = generic_gradient_magnitude(imf, sobel)
    elif feature == 'edge': 
        aFeature = np.double(sobel(ima) > w)
        bFeature = np.double(sobel(imb) > w)
        fFeature = np.double(sobel(imf) > w)
    elif feature == 'dct':  
        aFeature = dctn(ima, type=2, norm='ortho')
        bFeature = dctn(imb, type=2, norm='ortho')
        fFeature = dctn(imf, type=2, norm='ortho')
    elif feature == 'wavelet': 
        raise NotImplementedError('Wavelet feature extraction not yet implemented in Python!')
    else:
        raise ValueError(
            "Please specify a feature extraction method among 'gradient', 'edge', 'dct', 'wavelet', or 'none' (raw pixels)!")

    m, n = aFeature.shape
    w = w // 2
    fmi_map = np.ones((m - 2 * w, n - 2 * w))
    pass


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r


def fspecial_gaussian(shape, sigma):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gradient(x,flag):
    x = x.astype(np.float32)
    if flag == 1:
        y = np.concatenate([x, x], axis=1)[:,1:1 + x.shape[1]]
        return y - x
    elif flag == 3:
        y = np.concatenate([x, x], axis=0)[1:1 + x.shape[1],:]
        return y - x

if __name__  == '__main__':
    print('Hello World')
