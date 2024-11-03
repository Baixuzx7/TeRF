import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image        
import imageio.v3 as imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 
import json

"""               Grounding DINO              """
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

"""          Segment Anything Model           """
from segment_anything import build_sam,SamPredictor,SamAutomaticMaskGenerator

"""         Recognize Anything Model          """
from ram.models import ram # tag2text
from ram.inference import inference_ram
import torchvision.transforms as TS

"""        Large Language Model Meta AI       """
from llama import Llama

"""    Visible-Infrared Image Fusion Model    """
from fusion.model import RDN as FusionNetwork


class UnionLVM(object):
    def __init__(self,device):
        """             Hyper-parameters       for       Large       Vision        Model                     """

        self.device = device 
        self.config_file             = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.ram_checkpoint          = "/data/BaiXuYa/Pretrained_Model/RAM/ram_swin_large_14m.pth"
        self.grounded_checkpoint     = "/data/BaiXuYa/Pretrained_Model/GroundingDINO/groundingdino_swint_ogc.pth"
        self.sam_checkpoint          = "/data/BaiXuYa/Pretrained_Model/SAM/sam_vit_h_4b8939.pth"
        self.box_threshold           = 0.25
        self.text_threshold          = 0.2
        self.iou_threshold           = 0.5
        self.split                   = ','

        self.GroudingDINO_Model = self.load_model(self.config_file, self.grounded_checkpoint, self.device)
        self.GroudingDINO_Model.to(self.device)
        self.RAM_Model = ram(pretrained=self.ram_checkpoint,image_size=384,vit='swin_l')
        self.RAM_Model.eval()
        self.RAM_Model = self.RAM_Model.to(self.device)
        self.predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        self.mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=self.sam_checkpoint).to(self.device))

        self.normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform = TS.Compose([TS.Resize((384, 384)),TS.ToTensor(), self.normalize])

    def load_model(self,model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def load_image(self,image_pil):  # for inference_bbox
        image_pil = image_pil.convert('RGB')
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pt , _ = transform(image_pil, None)  # 3, h, w
        return image_pt


    def inference_tags(self,image_pil):
        image_pt = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            tag_tuple = inference_ram(image_pt, self.RAM_Model)
        tags=tag_tuple[0].replace(' |', ',')
        tags_chinese=tag_tuple[1].replace(' |', ',')
        """ tag_tuple[0] is English, tag_tuple[1] is Chinese"""
        tag_list = []
        string = tag_tuple[0] + ' |'
        start_of_search,end_of_search = 0,len(tag_tuple) - 1
        while start_of_search != end_of_search:
            hints_symbol = string.find('|',start_of_search)
            if hints_symbol == end_of_search:
                break
            else:
                tag_list.append(string[start_of_search:hints_symbol - 1])
                start_of_search = hints_symbol + 1 + 1 
            pass
        del tag_list[-1]

        return tags,tag_list 


    def inference_bbox(self, image, caption):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.GroudingDINO_Model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0] 
        boxes = outputs["pred_boxes"].cpu()[0]  
        logits.shape[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold 
        logits_filt = logits_filt[filt_mask] 
        boxes_filt = boxes_filt[filt_mask] 
        logits_filt.shape[0]

        tokenlizer = self.GroudingDINO_Model.tokenizer
        tokenized = tokenlizer(caption)

        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())
            pass
        """
            tags/caption contains several types of words, we need to exclude the words which are not nouns
            Thus nq <= len(tags,caption)        
        """
        return boxes_filt, torch.Tensor(scores), pred_phrases 
                                 
    def inference_masks(self,raw_image,boxes_filt,scores,pred_phrases):
        boxes_filt = boxes_filt.to(self.device)
        H,W = raw_image.shape[0],raw_image.shape[1]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H]).to(boxes_filt.device)
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        nms_idx = torchvision.ops.nms(boxes_filt.to(scores.device), scores, self.iou_threshold).cpu().numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        boxes_filt = boxes_filt.cpu()

        self.predictor.set_image(raw_image)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, raw_image.shape[:2]).to(self.device)
        if transformed_boxes.size(0) == 0:
            masks = torch.zeros([1,1,H,W])
            print('Segementation Region may not exist!')
        else:
            masks, _, _ = self.predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes.to(self.device),
                    multimask_output = False,
            )   

        return masks,pred_phrases
    
    def process(self,raw_image,text_prompt = None):
        """
            raw_image : type pil
            text_prompt : None  -----> detect all objects
                          '  '  -----> detect specific object
        """
        with torch.no_grad():
            if text_prompt == None:
                tags,tag_list = self.inference_tags(raw_image)
                boxes_filt, scores, pred_phrases = self.inference_bbox(self.load_image(raw_image),tags)
                masks,pred_phrases  = self.inference_masks(np.asarray(raw_image),boxes_filt,scores,pred_phrases)
                return masks,pred_phrases
            else:
                boxes_filt, scores, pred_phrases = self.inference_bbox(self.load_image(raw_image),text_prompt)
                masks = self.inference_masks(np.asarray(raw_image),boxes_filt,scores,pred_phrases)
                pass
                return masks 

    def generate_all_masks(self,raw_image):
        
        anns = self.mask_generator.generate(np.asarray(raw_image))
        if len(anns) == 0:
            return 
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        cnt = 0
        temp = sorted_anns[0]['segmentation']
        mask_folder = np.zeros([temp.shape[0]//1,temp.shape[1]//1,3])
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i] * m
            mask_folder = mask_folder + img
            cnt = cnt + 1
        return (mask_folder*255).astype(np.uint8)


    def generate_color_masks(self,masks,random_color=True):
        if random_color:
            color = np.random.random((1, 3))
        else:
            color = np.array([30/255, 144/255, 255/255])
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image
    

class UnionLLM(object):
    def __init__(self):        
        self.LLaMa_Model = Llama.build(
        ckpt_dir="/data/BaiXuYa/Pretrained_Model/LLaMA/7B/",
        tokenizer_path="/data/BaiXuYa/Pretrained_Model/LLaMA/llama-7b/tokenizer.model",
        max_seq_len=2048,
        max_batch_size=4,
        )
        self.prev_prompts_path = "./llama/prev_prompts.txt"
        self.prev_prompts = self.generate_prev_prompmts(self.prev_prompts_path)

    def generate_prev_prompmts(self,prev_prompts_path):
        pfiles = open(prev_prompts_path,"r").readlines()
        prev_prompts = ""
        for line in pfiles:
            prev_prompts = prev_prompts + "".join(line.split(","))
            pass
        return prev_prompts

    def generate_prompts(self,querries):
        prompts_list = []
        querries = querries.split(",")
        querry_list = []
        for querry in querries:
            querry_process = querry.strip()
            querry_list.append(querry_process)
            prompt = self.prev_prompts + querry_process + " =>\n        "
            prompts_list.append(prompt.lower())
        return querry_list,prompts_list

    def process(self,querries):
        querry_list,prompts_list = self.generate_prompts(querries)
        results_list = self.LLaMa_Model.text_completion(
        prompts_list,
        max_gen_len=128,
        temperature=0.6,
        top_p=0.9,
        )
        outputs_list = []
        for prompt, querry , result in zip(prompts_list, querry_list, results_list):
            output = result["generation"]
            cut_idx = output.find("\n")
            outputs_list.append(output[:cut_idx])
        return outputs_list
    
    def convert_dict(self,outputs):
        process_list = []
        for sentence in outputs:
            print(sentence) 
            if sentence == '':
                sentence = str('{"object" : "unknown" | "effect" : "unknown"}')
            t = json.loads(sentence.replace("|",",")) # dict
            keys_list = list(t.keys())
            t_reload = dict()
            t_reload["object"] = t[keys_list[0]]
            t_reload["effect"] = t[keys_list[1]]
            process_list.append(t_reload)
            pass

        return process_list


    def convert_task_object(self,outputs_dict):
        task_list,object_list = [], []
        for i in range(len(outputs_dict)):

            if outputs_dict[i]['effect'] == 'Denoise the region'.lower():
                task_list.append(1)
            elif outputs_dict[i]['effect'] == 'Enhanced illumination'.lower():
                task_list.append(2)
            elif outputs_dict[i]['effect'] == 'Blur the target'.lower():
                task_list.append(3)
            elif outputs_dict[i]['effect'] == 'Enhance the texture'.lower():
                task_list.append(4)
            elif outputs_dict[i]['effect'] == 'Make object more salient'.lower():
                task_list.append(5)
            elif outputs_dict[i]['effect'] == 'Reduced illumination'.lower():
                task_list.append(6)    
            else:
                task_list.append(0)
            object_list.append(outputs_dict[i]['object'])
        return object_list,task_list


class UnionFusion(object):
    def __init__(self,device = 'cuda'):
        self.device = device
        self.Fusion_Model = FusionNetwork().to(self.device)
        self.Fusion_Model.load_state_dict(torch.load(f'./fusion/checkpoint/net/Best.pth',map_location=self.device))
        self.transform_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])

    def process(self,image_vi,image_ir):
        image_vi_pt = self.transform_to_tensor(image_vi)
        image_ir_pt = self.transform_to_tensor(image_ir)
        with torch.no_grad():
            image_output = self.Fusion_Model(image_ir_pt.unsqueeze(0).to(self.device),
                                             image_vi_pt.unsqueeze(0).to(self.device))
        
        image_fusion = (self.tf2np(torch.clamp(image_output,max=1,min=0)) * 255).astype(np.uint8)
        return Image.fromarray(image_fusion, mode='RGB')

    def image_merge(self,image_Y_tf,image_YCrCb_tf):
        with torch.no_grad():
            image_merge_tf = image_YCrCb_tf
            image_merge_tf[:,0,:,:] = image_Y_tf
            image_YCrCb_np = (self.tf2np(torch.clamp(image_merge_tf,min=0.,max=1.)) * 255).astype(np.uint8)
            image_pil = cv2.cvtColor(image_YCrCb_np, cv2.COLOR_YCR_CB2RGB)
        return image_pil

    def tf2np(self,image_tf):
        n,c,h,w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        
        return image_np



if __name__ == '__main__':
    print('Hello World')
    