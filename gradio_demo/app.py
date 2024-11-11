import os
import cv2
import sys
sys.path.append('../')
import glob
import torch
import argparse
import numpy as np
import gradio as gr
from PIL import Image
from skimage import io
from typing import List
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
import apply_net
from utils_mask import get_mask_location
from diffusers import DDPMScheduler,AutoencoderKL
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.detectron2.projects.DensePose.apply_net_gradio import DensePose4Gradio
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Try-on Pipeline")
    parser.add_argument("--base_path", type=str, default='idm_filter_tunnel', help="Path to model checkpoint base directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
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


def start_tryon(dict,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    
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
        with torch.cuda.amp.autocast():
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
    
    
if __name__ == "__main__":
    
    args = parse_args()
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    densepose_model_hd = DensePose4Gradio(
    cfg='../preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml',
    model='https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl',
)
    example_path = os.path.join(os.path.dirname(__file__), 'example')
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
    
    garm_list = os.listdir(os.path.join(example_path,"cloth"))
    garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

    human_list = os.listdir(os.path.join(example_path,"human"))
    human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

    human_ex_list = []
    for ex_human in human_list_path:
        ex_dict= {}
        ex_dict['background'] = ex_human
        ex_dict['layers'] = None
        ex_dict['composite'] = None
        human_ex_list.append(ex_dict)

    ##default human
    js = """
    function createGradioAnimation() {
        // 웹페이지 전체에 배경 이미지 설정
        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.style.backgroundImage = "url('https://i.ibb.co/4sy7mGs/gradio-demo-3.png')";
        gradioContainer.style.backgroundSize = 'cover';
        gradioContainer.style.backgroundRepeat = 'repeat';
        gradioContainer.style.backgroundPosition = 'center';

        // 텍스트 애니메이션 컨테이너 설정
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = 'white'; // 배경과 구분되도록 텍스트 색상 설정

        var text = 'Welcome to SKT Fellowship 6기!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.5s';
                    letter.innerText = text[i];

                    container.appendChild(letter);

                    setTimeout(function() {
                        letter.style.opacity = '1';
                    }, 50);
                }, i * 250);
            })(i);
        }

        gradioContainer.insertBefore(container, gradioContainer.firstChild);

        return 'Animation created';
    }

    """


    image_blocks = gr.Blocks(js=js).queue(max_size=1)

    # image_blocks = gr.Blocks(css=".gradio-container {background: url('file=https://i.ibb.co/qjD45Z7/gradio-demo.png')}").queue(max_size=4)

    with image_blocks as demo:
        gr.Markdown(
            "<span style='color: white; font-size: 24px; font-weight: bold;'>⛳🥇🎀 Louis VTON 👕👔👚</span>",
        )
        gr.Markdown(
            "<span style='color: white; font-size: 16px; font-weight: bold;'>Virtual Try-on with your image and garment image.</span>",
        )
        # gr.Markdown("Virtual Try-on with your image and garment image.")
        with gr.Row():
            with gr.Column():
                imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
                with gr.Row():
                    is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
                with gr.Row():
                    is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)

                example = gr.Examples(
                    inputs=imgs,
                    examples_per_page=4,
                    examples=human_ex_list
                )

            with gr.Column():
                garm_img = gr.Image(label="Garment", sources='upload', type="pil")
                with gr.Row(elem_id="prompt-container"):
                    with gr.Row():
                        prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
                example = gr.Examples(
                    inputs=garm_img,
                    examples_per_page=5,
                    examples=garm_list_path)
            with gr.Column():
                # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
            with gr.Column():
                # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)




        with gr.Column():
            try_button = gr.Button(value="Try-on")
            with gr.Accordion(label="Advanced Settings", open=False):
                with gr.Row():
                    denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                    seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)



        try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')




    image_blocks.launch(share=True)

