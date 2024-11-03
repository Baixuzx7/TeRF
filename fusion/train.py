
import os
import shutil 

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import torch
import numpy as np
import random
from tqdm import tqdm
from configs import BaseOptions
from model import RDN as FusionNetwork
from model import FusionTrainer
from dataset import VIR_NIR_Dataset

from metrics import analysis_AG,analysis_CC,analysis_EN,analysis_PSNR,analysis_ssim,analysis_VIF,analysis_SCD,analysis_SF,analysis_SD,analysis_MI


def train(configs):
    train_dataset = VIR_NIR_Dataset(configs.opt.train_root,istrain=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.opt.batch_size, shuffle=configs.opt.isshuffle)
    valid_dataset = VIR_NIR_Dataset(configs.opt.valid_root,istrain=False)
    device = torch.device('cuda:{}'.format(configs.opt.gpu_id) if torch.cuda.is_available() else 'cpu')
    fusion_model = FusionNetwork()
    trainer = FusionTrainer(fusion_model, opt = configs.opt)
    PSNR_MIN = 0
    for epoch in range(configs.opt.n_epochs):
        """ Train """
        trainer.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for (image_ir,image_vi_Y,image_vi,image_VI_YCrCb) in tepoch:
                """ trainer """
                trainer.optimizer.zero_grad()
                fusion_loss = trainer(image_ir, image_vi)
                fusion_loss.backward()
                trainer.optimizer.step()
                tepoch.set_description_str('Epoch:{}/{}  fusion_loss: {:.6f} '.format(epoch, configs.opt.n_epochs,fusion_loss.item()))
                pass
        trainer.scheduler.step()
        """ Valid """ 
        trainer.eval()
        if (epoch > -1) and (epoch % configs.opt.save_per_epoch == 0): 
            AG_list,CC_list,EN_list,PSNR_list,ssim_list,VIF_list,SCD_list,SF_list,SD_list,MI_list = [],[],[],[],[],[],[],[],[],[]
            for j in range(len(valid_dataset)):
                image_ir,image_vi_Y,image_vi,image_VI_YCrCb = valid_dataset[j]
                with torch.no_grad():
                    image_output = trainer.fusion_model(image_ir.unsqueeze(0).to(trainer.device),image_vi.unsqueeze(0).to(trainer.device))
                    
                    image_fn = trainer.tf2img(image_output.to(trainer.device))
                    image_ir = trainer.tf2img(image_ir.unsqueeze(0))
                    image_vi = trainer.tf2img(image_vi.unsqueeze(0))
 
                    AG_list.append(analysis_AG(image_fn))
                    CC_list.append(analysis_CC(image_vi,image_ir,image_fn))
                    EN_list.append(analysis_EN(image_fn))
                    PSNR_list.append(analysis_PSNR(image_vi,image_ir,image_fn))
                    ssim_list.append(analysis_ssim(image_vi,image_ir,image_fn))
                    # VIF_list.append(analysis_VIF(image_vi,image_ir,image_fn))
                    # SCD_list.append(analysis_SCD(image_vi,image_ir,image_fn))
                    # SF_list.append(analysis_SF(image_fn))
                    # SD_list.append(analysis_SD(image_fn))
                    MI_list.append(analysis_MI(image_vi,image_ir,image_fn))

            AG_mean,AG_var = np.asarray(AG_list).mean(),np.asarray(AG_list).var()
            CC_mean,CC_var = np.asarray(CC_list).mean(),np.asarray(CC_list).var()
            EN_mean,EN_var = np.asarray(EN_list).mean(),np.asarray(EN_list).var()
            PSNR_mean,PSNR_var = np.asarray(PSNR_list).mean(),np.asarray(PSNR_list).var()
            SSIM_mean,SSIM_var = np.asarray(ssim_list).mean(),np.asarray(ssim_list).var()
            VIF_mean,VIF_var = 0,0 #np.asarray(VIF_list).mean(),np.asarray(VIF_list).var()
            SCD_mean,SCD_var = 0,0 #np.asarray(SCD_list).mean(),np.asarray(SCD_list).var()
            SF_mean,SF_var = 0,0 #np.asarray(SF_list).mean(),np.asarray(SF_list).var()
            SD_mean,SD_var = 0,0 #np.asarray(SD_list).mean(),np.asarray(SD_list).var()
            MI_mean,MI_var = np.asarray(MI_list).mean(),np.asarray(MI_list).var()

            print('Evaluate mean ---> AG :  {:.4f} CC :  {:.4f} EN :  {:.4f} PSNR :  {:.4f} SSIM :  {:.4f} VIF :  {:.4f} SCD :  {:.4f} SF :  {:.4f} SD :  {:.4f} MI :  {:.4f}     '.format(
                AG_mean,CC_mean,EN_mean,PSNR_mean,SSIM_mean,VIF_mean,SCD_mean,SF_mean,SD_mean,MI_mean))

            print('Evaluate  var ---> AG :  {:.4f} CC :  {:.4f} EN :  {:.4f} PSNR :  {:.4f} SSIM :  {:.4f} VIF :  {:.4f} SCD :  {:.4f} SF :  {:.4f} SD :  {:.4f} MI :  {:.4f}     '.format(
                AG_var,CC_var,EN_var,PSNR_var,SSIM_var,VIF_var,SCD_var,SF_var,SD_var,MI_var))

            # print('LaTex content ---> {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} & {:.4f}$\pm${:.4f} \\\\ '.format(
            #     AG_mean,AG_var,
            #     CC_mean,CC_var,
            #     EN_mean,EN_var,
            #     PSNR_mean,PSNR_var,
            #     SSIM_mean,SSIM_var,
            #     VIF_mean,VIF_var,
            #     SCD_mean,SCD_var,
            #     SF_mean,SF_var,
            #     SD_mean,SD_var))

            trainer.writer.add_scalar('AG_mean',AG_mean,global_step=epoch)
            trainer.writer.add_scalar('CC_mean' ,CC_mean,global_step=epoch)
            trainer.writer.add_scalar('EN_mean' ,EN_mean,global_step=epoch)
            trainer.writer.add_scalar('PSNR_mean',PSNR_mean,global_step=epoch)
            trainer.writer.add_scalar('SSIM_mean' ,SSIM_mean,global_step=epoch)
            trainer.writer.add_scalar('VIF_mean' ,VIF_mean,global_step=epoch)
            trainer.writer.add_scalar('SCD_mean' ,SCD_mean,global_step=epoch)
            trainer.writer.add_scalar('SF_mean' ,SF_mean,global_step=epoch)
            trainer.writer.add_scalar('SD_mean' ,SD_mean,global_step=epoch)
        
            """ SAVE BEST Parameters"""
            if PSNR_mean > PSNR_MIN:
                PSNR_MIN = PSNR_mean
                trainer.save_model_parameters(trainer.fusion_model,configs.opt.net_params_dir,'best')
                trainer.save_optim_parameters(trainer.optimizer,configs.opt.opt_params_dir,'best')
                pass
        """ trainer SAVE Parameters """
        trainer.save_model_parameters(trainer.fusion_model,configs.opt.net_params_dir,epoch)
        trainer.save_optim_parameters(trainer.optimizer,configs.opt.opt_params_dir,epoch)

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
        if os.path.exists('./checkpoint'):
            shutil.rmtree('./checkpoint')
        if os.path.exists('./result'):
            shutil.rmtree('./result')
        if os.path.exists('./blog'):
            shutil.rmtree('./blog')
        pass
    """   Train  """
    train(configs)
    """   Test  """
    # test(configs)
