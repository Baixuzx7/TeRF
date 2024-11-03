
import os
import imageio.v3 as imageio
import numpy as np
from metrics import analysis_AG,analysis_CC,analysis_EN,analysis_PSNR,analysis_ssim,analysis_VIF,analysis_SCD,analysis_SF,analysis_SD,analysis_MI,analysis_Qabf

output_save_path = "./fusion_metric/"
if not os.path.exists(output_save_path):
    os.makedirs(output_save_path)

print('Start training.....')

""" Eval """
ir_file_dir = './data/vision/RoadScene/ir/'
vi_file_dir = './data/vision/RoadScene/vi/'
fn_file_dir = './outputs/results/'
ir_file_list = os.listdir(ir_file_dir)
ir_file_list.sort(key=lambda x: int(x[:-4]))
AG_list,CC_list,EN_list,PSNR_list,ssim_list,VIF_list,SCD_list,SF_list,SD_list,MI_list,Qabf_list = [],[],[],[],[],[],[],[],[],[],[]

for j in range(len(ir_file_list)):
    print(j,'/',len(ir_file_list))
    image_vi = imageio.imread(vi_file_dir + str(j) + '.png')
    image_ir = imageio.imread(ir_file_dir + str(j) + '.png') 
    image_fn = imageio.imread(fn_file_dir + str(j) + '.png')

    AG_list.append(analysis_AG(image_fn))
    CC_list.append(analysis_CC(image_vi,image_ir,image_fn))
    EN_list.append(analysis_EN(image_fn))
    PSNR_list.append(analysis_PSNR(image_vi,image_ir,image_fn))
    ssim_list.append(analysis_ssim(image_vi,image_ir,image_fn))
    VIF_list.append(analysis_VIF(image_vi,image_ir,image_fn))
    SCD_list.append(analysis_SCD(image_vi,image_ir,image_fn))
    SF_list.append(analysis_SF(image_fn))
    SD_list.append(analysis_SD(image_fn))
    MI_list.append(analysis_MI(image_vi,image_ir,image_fn))
    Qabf_list.append(analysis_Qabf(image_vi,image_ir,image_fn))

AG_mean,AG_var = np.asarray(AG_list).mean(),np.asarray(AG_list).var()
CC_mean,CC_var = np.asarray(CC_list).mean(),np.asarray(CC_list).var()
EN_mean,EN_var = np.asarray(EN_list).mean(),np.asarray(EN_list).var()
PSNR_mean,PSNR_var = np.asarray(PSNR_list).mean(),np.asarray(PSNR_list).var()
SSIM_mean,SSIM_var = np.asarray(ssim_list).mean(),np.asarray(ssim_list).var()
VIF_mean,VIF_var = np.asarray(VIF_list).mean(),np.asarray(VIF_list).var()
SCD_mean,SCD_var = np.asarray(SCD_list).mean(),np.asarray(SCD_list).var()
SF_mean,SF_var = np.asarray(SF_list).mean(),np.asarray(SF_list).var()
SD_mean,SD_var = np.asarray(SD_list).mean(),np.asarray(SD_list).var()
MI_mean,MI_var = np.asarray(MI_list).mean(),np.asarray(MI_list).var()
Qabf_mean,Qabf_var = np.asarray(Qabf_list).mean(),np.asarray(Qabf_list).var()

np.save(output_save_path + "AG.npy",np.asarray(AG_list))
np.save(output_save_path + "CC.npy",np.asarray(CC_list))
np.save(output_save_path + "EN.npy",np.asarray(EN_list))
np.save(output_save_path + "PSNR.npy",np.asarray(PSNR_list))
np.save(output_save_path + "SSIM.npy",np.asarray(ssim_list))
np.save(output_save_path + "VIF.npy",np.asarray(VIF_list))
np.save(output_save_path + "SCD.npy",np.asarray(SCD_list))
np.save(output_save_path + "SF.npy",np.asarray(SF_list))
np.save(output_save_path + "SD.npy",np.asarray(SD_list))
np.save(output_save_path + "MI.npy",np.asarray(MI_list))
np.save(output_save_path + "Qabf.npy",np.asarray(Qabf_list))

print('Finished')