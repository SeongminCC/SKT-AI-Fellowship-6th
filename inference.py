import sys
sys.path.append('./')
import os
import cv2
import json
import glob
import time
import torch
import argparse
import numpy as np
from PIL import Image
from skimage import io
from typing import List
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import AutoTokenizer
from diffusers import DDPMScheduler,AutoencoderKL
from torchvision.transforms.functional import to_pil_image
from gradio_demo.apply_net import *
from gradio_demo.utils_mask import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.detectron2.projects.DensePose.apply_net_gradio import DensePose4Gradio
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

### 가중치 경로 선택 ###
# base_path = 'idm'  # pre-trained
# base_path = 'idm_filter/checkpoint-100'  # filtering
# base_path = 'idm_filter_tunnel/checkpoint-40'  # tunnel

def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Try-on Pipeline")
    parser.add_argument("--base_path", required=True, help="Path to model checkpoint base directory")
    parser.add_argument("--device", type=str, default="0", help="Device to use (e.g., cuda:0 or cpu)")
    parser.add_argument("--denoise_steps", type=int, default=30, help="Number of denoise steps for the pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input_dir", type=str, default='Input')
    parser.add_argument("--output_dir", type=str, default='Output')
    args = parser.parse_args()
    return args



# PIL 이미지를 binary mask로 변환
def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


### filtering
#### 이미지에 엣지 필터 적용
def edge_filter(img, filter_name):
    # PIL 이미지 객체를 numpy 배열로 변환
    img = np.array(img.convert('L'))  # 그레이스케일로 변환

    # sobel filter 적용
    if filter_name == "sobel":
        sobel_x = cv2.Sobel(img, cv2.CV_6장
    os.makedirs(os.path.join(output_dir, base_path), exist_ok=True)
    output_img_save_path = os.path.join(output_dir, base_path, f"{img_name}_{gar_name}_output.png")
    output_img.save(output_img_save_path)
    
    
if __name__ == "__main__":
    
    args = parse_args()
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Densepose 모델 초기화
    densepose_model_hd = DensePose4Gradio(
        cfg='preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        model='https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    )

    # Pretrained 모델 로드드
    vae = AutoencoderKL.from_pretrained(args.base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
        )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
 장
    for i, human_img_path in enumerate(human_list):
        for j, garm_img_path in enumerate(garm_list): 
            
            start_time = time.time()
            save_result(human_img_path, garm_img_path, garm_prom_dict, args.base_path, args.output_dir, args.denoise_steps, args.seed)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"코드 실행 시간: {execution_time:.5f}초")
