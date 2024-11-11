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
def edge_filter(img, filter_name):
    # PIL 이미지 객체를 numpy 배열로 변환
    img = np.array(img.convert('L'))  # 그레이스케일로 변환

    if filter_name == "sobel":
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.hypot(sobel_x, sobel_y)  # 그라디언트의 크기 계산
        sobel_normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(sobel_normalized)

    if filter_name == "canny":
        edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)
        return Image.fromarray(edges)

    if filter_name == "laplacian":
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(laplacian_normalized)

    if filter_name == "gabor":
        gabor_kernel = cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D((img * 255).astype(np.uint8), cv2.CV_8UC3, gabor_kernel)
        return Image.fromarray(filtered_img)

    
def start_tryon(human_img, garm_img, garm_path, garment_des, is_checked, is_checked_crop, denoise_steps, seed, base_path):
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    fil_cloth = edge_filter(img=garm_img, filter_name="sobel")
    
    human_img_orig = human_img["background"].convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size   # 768, 1024
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(human_img["layers"][0].convert("RGB").resize((768, 1024)))
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    pose_img = densepose_model_hd.execute(human_img_arg)
    pose_img = np.array(pose_img)[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        prompt = "model is wearing " + garment_des
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt = "a photo of " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            if not isinstance(prompt, List):
                prompt = [prompt] * 1
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * 1
            with torch.inference_mode():
                (
                    prompt_embeds_c,
                    _,
                    _,
                    _,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )

            pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16) # ([1, 3, 1024, 768])
            fil_cloth_tensor = tensor_transform(fil_cloth).unsqueeze(0).to(device, torch.float16) # ([1, 1, 1024, 768])
           
        
            if args.base_path.split('/')[-1] == 'idm':
                cloth_filtering = False
                cloth_logo_masking = False
            else:
                cloth_filtering = True
                cloth_logo_masking = False
            
            generator = torch.Generator(device).manual_seed(args.seed) if args.seed is not None else None
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=args.denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img.to(device, torch.float16),
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor.to(device, torch.float16),
                cloth_filtering=cloth_filtering, # filtering 유무 설정
                cloth_filter=fil_cloth_tensor.to(device, torch.float16),
                cloth_logo_masking=False, # logo_mask 유무 설정
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
                        
def save_result(human_img_path, garm_img_path, garm_prom_dict, base_path, output_dir, denoise_steps, seed):
    human_img = Image.open(human_img_path)
    garm_img = Image.open(garm_img_path)
    
    img_name = human_img_path.split('/')[-1].split('.')[0]
    gar_name = garm_img_path.split('/')[-1].split('.')[0]
    # img_name = '_'.join(human_img_path.split('/')[-1].split('_')[2:4])[:-4]
    # gar_name = ''.join(garm_img_path.split('/')[-1].split('_')[2:4])[:-4]
    
    garment_des = garm_prom_dict[garm_img_path.split('/')[-1]]

    is_checked = True  # Use auto-generated mask
    is_checked_crop = True  # Use auto-crop & resizing
    
    human_img_dict = {
        "background": human_img,
        "layers": [human_img]  # Assuming 'layers' is just the human image in this context
    }
    
    output_img, masked_img = start_tryon(human_img_dict, garm_img, garm_img_path, garment_des, is_checked, is_checked_crop, denoise_steps, seed, base_path)
    

    os.makedirs(os.path.join(output_dir, base_path), exist_ok=True)
    output_img_save_path = os.path.join(output_dir, base_path, f"{img_name}_{gar_name}_output.png")
    output_img.save(output_img_save_path)
    
    
if __name__ == "__main__":
    
    args = parse_args()
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    densepose_model_hd = DensePose4Gradio(
        cfg='preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        model='https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
    )
    
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
    )
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        args.base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_path, subfolder="scheduler")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    tensor_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
        )

    pipe = TryonPipeline.from_pretrained(
            args.base_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    
    file_list = sorted(glob.glob(f'{args.input_dir}/*/*'))
    garm_list = [file for file in file_list if 'garment' in file]
    human_list = [file for file in file_list if 'image' in file]
    with open('Input/garments.json', 'r') as json_file:
        garm_prom_dict = json.load(json_file)
        
    # desired_pairs = [(0, 0), (0, 3), (1, 1), (1, 4), (2, 2)]
    for i, human_img_path in enumerate(human_list):
        for j, garm_img_path in enumerate(garm_list): 
            
            start_time = time.time()
            save_result(human_img_path, garm_img_path, garm_prom_dict, args.base_path, args.output_dir, args.denoise_steps, args.seed)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"코드 실행 시간: {execution_time:.5f}초")