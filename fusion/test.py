import torch
import numpy as np
import random
from tqdm import tqdm
from configs import BaseOptions
from model import RDN as FusionNetwork
from model import FusionTrainer
from dataset import VIR_NIR_Dataset

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from metrics import analysis_AG,analysis_CC,analysis_EN,analysis_PSNR,analysis_ssim,analysis_VIF,analysis_SCD,analysis_SF,analysis_SD,analysis_MI
import os
import shutil
import imageio

from openpyxl import Workbook

import pandas as pd

def test(configs):
    test_dataset = VIR_NIR_Dataset(configs.opt.test_root,istrain = False)
    device = torch.device('cuda:{}'.format(configs.opt.gpu_id) if torch.cuda.is_available() else 'cpu')
    fusion_model = FusionNetwork().to(device)
    trainer = FusionTrainer(fusion_model, opt = configs.opt)
    trainer.load_model_parameters(trainer.fusion_model,configs.opt.net_params_dir,'299')
    """ Test """
    AG_list,CC_list,EN_list,PSNR_list,ssim_list,VIF_list,SCD_list,SF_list,SD_list,MI_list = [],[],[],[],[],[],[],[],[],[]
    trainer.eval()
    for idx in range(len(test_dataset)):
        image_ir,image_vi_Y,image_vi,image_VI_YCrCb = test_dataset[idx]
        with torch.no_grad():
            print(idx,'  / ',len(test_dataset))
            image_output = trainer.fusion_model(image_ir.unsqueeze(0).to(trainer.device),image_vi.unsqueeze(0).to(trainer.device))
            
            image_fn = trainer.tf2img(image_output.to(trainer.device))
            image_ir = trainer.tf2img(image_ir.unsqueeze(0))
            image_vi = trainer.tf2img(image_vi.unsqueeze(0))
 
            
            imageio.imwrite('./fusion/result/fusion/' + str(idx) + '.png',image_fn.astype(np.uint8))
            imageio.imwrite('./fusion/result/ir/' + str(idx) + '.png',image_ir.astype(np.uint8))
            imageio.imwrite('./fusion/result/vi/' + str(idx) + '.png',image_vi.astype(np.uint8))

            # image_save = np.concatenate([image_ir,image_vi,image_fusion],axis=1)
            # trainer.save_image(image=image_save,save_path=trainer.opt.result_dir,name=idx)
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


    print('Evaluate mean ---> AG :  {:.4f} CC :  {:.4f} EN :  {:.4f} PSNR :  {:.4f} SSIM :  {:.4f} VIF :  {:.4f} SCD :  {:.4f} SF :  {:.4f} SD :  {:.4f} MI :  {:.4f}   '.format(
        AG_mean,CC_mean,EN_mean,PSNR_mean,SSIM_mean,VIF_mean,SCD_mean,SF_mean,SD_mean,MI_mean))

    print('Evaluate  var ---> AG :  {:.4f} CC :  {:.4f} EN :  {:.4f} PSNR :  {:.4f} SSIM :  {:.4f} VIF :  {:.4f} SCD :  {:.4f} SF :  {:.4f} SD :  {:.4f} MI :  {:.4f}   '.format(
        AG_var,CC_var,EN_var,PSNR_var,SSIM_var,VIF_var,SCD_var,SF_var,SD_var,MI_var))

    print('LaTex content ---> {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} \\\\ '.format(
        AG_mean,AG_var,
        CC_mean,CC_var,
        EN_mean,EN_var,
        PSNR_mean,PSNR_var,
        SSIM_mean,SSIM_var,
        VIF_mean,VIF_var,
        SCD_mean,SCD_var,
        SF_mean,SF_var,
        SD_mean,SD_var,
        MI_mean,MI_var))



    np.save("./fusion/result/metric/AG.npy",np.asarray(AG_list))
    np.save("./fusion/result/metric/CC.npy",np.asarray(CC_list))
    np.save("./fusion/result/metric/EN.npy",np.asarray(EN_list))
    np.save("./fusion/result/metric/PSNR.npy",np.asarray(PSNR_list))
    np.save("./fusion/result/metric/SSIM.npy",np.asarray(ssim_list))
    np.save("./fusion/result/metric/VIF.npy",np.asarray(VIF_list))
    np.save("./fusion/result/metric/SCD.npy",np.asarray(SCD_list))
    np.save("./fusion/result/metric/SF.npy",np.asarray(SF_list))
    np.save("./fusion/result/metric/SD.npy",np.asarray(SD_list))
    np.save("./fusion/result/metric/MI.npy",np.asarray(MI_list))

    data_save = np.zeros([len(AG_list),10])

    data_save[:,0] = np.asarray(AG_list)
    data_save[:,1] = np.asarray(CC_list)
    data_save[:,2] = np.asarray(EN_list)
    data_save[:,3] = np.asarray(PSNR_list)
    data_save[:,4] = np.asarray(ssim_list)
    data_save[:,5] = np.asarray(VIF_list)
    data_save[:,6] = np.asarray(SCD_list)
    data_save[:,7] = np.asarray(SF_list)
    data_save[:,8] = np.asarray(SD_list)
    data_save[:,9] = np.asarray(MI_list)
    

    # 加载npy文件 
    data_df=pd.DataFrame(data_save)
    # 创建一个新的工作簿 
    
    writer = pd.ExcelWriter('metric.xlsx')  # 生成一个excel文件
    data_df.to_excel(writer,'page_1')  # 数据写入excel文件
    writer._save()
    print('Finished')


""" Seed Setting"""
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    """   Setting Parameters   """
    setup_seed(45)
    torch.cuda.empty_cache()
    configs = BaseOptions()
    configs.print_options()
    configs.initialize()
    # """   Romve the abundant files """
    if configs.opt.iscontinue is False:
        if os.path.exists('./fusion/result'):
            shutil.rmtree('./fusion/result')
        pass
    output_save_path = "./fusion/result"
    if not os.path.exists(output_save_path + '/metric'):
        os.makedirs(output_save_path + '/metric')
    if not os.path.exists(output_save_path + '/fusion'):
        os.makedirs(output_save_path + '/fusion')
    if not os.path.exists(output_save_path + '/ir'):
        os.makedirs(output_save_path + '/ir')
    if not os.path.exists(output_save_path + '/vi'):
        os.makedirs(output_save_path + '/vi')

    """   Train  """
    # train(configs)
    """   Test  """
    test(configs)
