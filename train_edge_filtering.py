import os
import random
import argparse
import json
import itertools
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage import io
from PIL import Image
import cv2
import numpy as np
from transformers import CLIPImageProcessor, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

from ip_adapter.ip_adapter import Resampler
from diffusers.utils.import_utils import is_xformers_available
from typing import Literal, Tuple,List
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF

os.environ['NCCL_P2P_LEVEL'] = 'NVL'
# os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_SOCKET_IFNAME'] = 'enp3s0f1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


# 지정한 필터를 통해 garment image로부터 edge 정보를 담은 binary image 생성
def edge_filter(img_path, filter_name):
    img = io.imread(img_path, as_gray=True)

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

    

# 커스텀 데이터셋 클래스 정의의
class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size


        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()

        with open(
            os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            # "colors",
            # "textures",
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str


        self.order = order

        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.clip_processor = CLIPImageProcessor()
        
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        # subject_txt = self.txt_preprocess['train']("shirt")
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))
        fil_cloth = edge_filter(img_path=os.path.join(self.dataroot, self.phase, "cloth", c_name), filter_name="sobel")

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))

        image = self.transform(im_pil_big)
        # load parsing image


        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask", im_name.replace('.jpg','_mask.png'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        densepose_name = im_name
        densepose_map = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", densepose_name)
        )
        pose_img = self.toTensor(densepose_map)  # [-1,1]
 
 

        if self.phase == "train":
            if random.random() > 0.5: # 랜덤 좌우 반전
                cloth = self.flip_transform(cloth)
                fil_cloth = self.flip_transform(fil_cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)



            if random.random()>0.5: # 랜덤 색상 변화
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
                
                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

              
            if random.random() > 0.5: # 랜덤 스케일 조정
                scale_val = random.uniform(0.8, 1.2)
                image = transforms.functional.affine(
                    image, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                mask = transforms.functional.affine(
                    mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )
                pose_img = transforms.functional.affine(
                    pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )



            if random.random() > 0.5: # 랜덤 이동
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                image = transforms.functional.affine(
                    image,
                    angle=0,
                    translate=[shift_valx * image.shape[-1], shift_valy * image.shape[-2]],
                    scale=1,
                    shear=0,
                )
                mask = transforms.functional.affine(
                    mask,
                    angle=0,
                    translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]],
                    scale=1,
                    shear=0,
                )
                pose_img = transforms.functional.affine(
                    pose_img,
                    angle=0,
                    translate=[
                        shift_valx * pose_img.shape[-1],
                        shift_valy * pose_img.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )




        mask = 1-mask

        cloth_trim =  self.clip_processor(images=cloth, return_tensors="pt").pixel_values


        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        im_mask = image * mask

        pose_img =  self.norm(pose_img)


        result = {}
        result["c_name"] = c_name
        result["image"] = image
        result["cloth"] = cloth_trim
        result["cloth_pure"] = self.transform(cloth)
        result["cloth_filter"] = self.transform(fil_cloth)
        result["inpaint_mask"] = 1-mask
        result["im_mask"] = im_mask
        result["caption"] = "model is wearing " + cloth_annotation
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["annotation"] = cloth_annotation
        result["pose_img"] = pose_img


        return result

    def __len__(self):
        return len(self.im_names)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--pretrained_garmentnet_path",type=str,default="stabilityai/stable-diffusion-xl-base-1.0",required=False,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    
    # checkpoint
    parser.add_argument("--checkpointing_epoch",type=int,default=1,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--pretrained_ip_adapter_path",type=str,default="ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",)
    parser.add_argument("--image_encoder_path",type=str,default="ckpt/image_encoder",required=False,help="Path to CLIP image encoder",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--logging_steps",type=int,default=1000,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--output_dir",type=str,default="output",help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--snr_gamma",type=float,default=None,help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--num_tokens",type=int,default=16,help=("IP adapter token nums"),)
    parser.add_argument("--learning_rate",type=float,default=1e-6,help="Learning rate to use.",)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=12, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    
    # epochs
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--guidance_scale",type=float,default=2.5,)
    parser.add_argument("--seed", type=int, default=42,)    
    parser.add_argument("--num_inference_steps",type=int,default=30,)    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--data_dir", type=str, default="/home/omnious/workspace/yisol/Dataset/VITON-HD/zalando", help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args





def main():
    
    
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        # mixed_precision=args.mixed_precision,
        mixed_precision="no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )
    

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # pre-trained model 정의의
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        # torch_dtype=torch.float32,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        # torch_dtype=torch.float16,
    )
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet_encoder",
        # torch_dtype=torch.float16,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        # torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        # torch_dtype=torch.float16,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    image_proj_model = unet.encoder_hid_proj
    image_proj_model.to(accelerator.device, dtype=torch.float32)
    
    image_proj_model.requires_grad_(True)

    # 인코딩된 garment image를 받는 new conv층 정의
    conv_new = torch.nn.Conv2d(
        in_channels=4+1, # channel of encoding garment image : 4 / binary image : 1
        out_channels=UNet_Encoder.conv_in.out_channels,
        kernel_size=3,
        padding=1,
    )

    torch.nn.init.kaiming_normal_(conv_new.weight)  
    conv_new.weight.data = conv_new.weight.data * 0. 
    conv_new.weight.data[:, :4] = UNet_Encoder.conv_in.weight.data  
    conv_new.bias.data = UNet_Encoder.conv_in.bias.data 

    UNet_Encoder.conv_in = conv_new  # replace conv layer in unet
    UNet_Encoder.config['in_channels'] = 5  # update config
    UNet_Encoder.config.in_channels = 5  # update config
    
    weight_dtype = torch.float32
    unet.to(accelerator.device, weight_dtype)
    vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device, weight_dtype)
    text_encoder_two.to(accelerator.device, weight_dtype)
    image_encoder.to(accelerator.device, weight_dtype)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    UNet_Encoder.requires_grad_(True) # image encoder만 학습습
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        UNet_Encoder.enable_gradient_checkpointing()

    UNet_Encoder.train()


    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_opt = itertools.chain(UNet_Encoder.parameters())

    optimizer = optimizer_class(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = VitonHDDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="paired",
        size=(args.height, args.width),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=16,
    )
    # test_dataset = VitonHDDataset(
    #     dataroot_path=args.data_dir,
    #     phase="test",
    #     order="paired",
    #     size=(args.height, args.width),
    # )
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     shuffle=False,
    #     batch_size=args.test_batch_size,
    #     num_workers=4,
    # )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)


    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch # 총 학습 step 수
        overrode_max_train_steps = True


    UNet_Encoder, optimizer,train_dataloader = accelerator.prepare(UNet_Encoder, optimizer, train_dataloader)
    initial_global_step = 0
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    progress_bar = tqdm(
        range(0, args.max_train_steps), # 총 train step 수
        initial=initial_global_step,
        desc="Steps", # 진행 막대 표시될 설명
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    first_epoch = 0
    train_loss = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(UNet_Encoder), accelerator.accumulate(image_proj_model):

                ### TryonNet ###
                # image -> vae : 잠재벡터 추출
                pixel_values = batch["image"].to(dtype=vae.dtype) # ([24, 3, 1024, 768])
                model_input = vae.encode(pixel_values).latent_dist.sample() # ([24, 4, 128, 96])
                model_input = model_input * vae.config.scaling_factor # ([24, 4, 128, 96])

                # im_mask -> vae : 잠재벡터 추출
                masked_latents = vae.encode(
                    batch["im_mask"].reshape(batch["image"].shape).to(dtype=vae.dtype) # ([24, 3, 1024, 768]) -> ([24, 4, 128, 96])
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor # ([24, 4, 128, 96])

                # latent와 caoncate를 하기 위해 resizing
                masks = batch["inpaint_mask"] # ([24, 1, 1024, 768])
                # resize the mask to latents shape as we concatenate the mask to the latents
                mask = torch.stack(
                    [
                        torch.nn.functional.interpolate(masks, size=(args.height // 8, args.width // 8)) # ([1, 24, 1, 128, 96])
                    ]
                )
                mask = mask.reshape(-1, 1, args.height // 8, args.width // 8) # ([24, 1, 128, 96])
                mask = mask.to(dtype = torch.float16)

                # pose_img -> vae : 잠재벡터 추출
                pose_map = vae.encode(batch["pose_img"].to(dtype=vae.dtype)).latent_dist.sample() # ([24, 3, 1024, 768]) -> ([24, 4, 128, 96])
                pose_map = pose_map * vae.config.scaling_factor # ([24, 4, 128, 96])

                # latent model image에 noise 추가
                noise = torch.randn_like(model_input)

                bsz = model_input.shape[0]
                timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(model_input, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask,masked_latents,pose_map], dim=1) # ([24, 13, 128, 96])

                # 2개의 tokenizer 정의
                text_input_ids = tokenizer_one(
                    batch['caption'],
                    max_length=tokenizer_one.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_two(
                    batch['caption'],
                    max_length=tokenizer_two.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                # # text 토큰화 처리 완료
                # with torch.no_grad():
                encoder_output = text_encoder_one(text_input_ids.to(accelerator.device), output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]

                encoder_output_2 = text_encoder_two(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]

                encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat


                def compute_time_ids(original_size, crops_coords_top_left = (0,0)):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.height, args.height) 
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device)
                    return add_time_ids

                # 배치 내 각 이미지에 대해 time ID를 계산하여 리스트에 저장
                add_time_ids = torch.cat(
                    [compute_time_ids((args.height, args.height)) for i in range(bsz)] # bsz : batch size
                )

                # 
                img_emb_list = []
                for i in range(bsz):
                    img_emb_list.append(batch['cloth'][i])

                image_embeds = torch.cat(img_emb_list,dim=0)
                image_embeds = image_encoder(image_embeds, output_hidden_states=True).hidden_states[-2]
                
                ip_tokens =image_proj_model(image_embeds)
                # ip_tokens = unet.encoder_hid_proj(image_embeds.to(accelerator.device, dtype=weight_dtype)).to(accelerator.device, dtype=weight_dtype)


                # add cond
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                unet_added_cond_kwargs["image_embeds"] = ip_tokens

                ## Question: ori_ver(cloth -> encoder) / pre_ver(cloth -> vae)
                cloth_values = batch["cloth_pure"].to(accelerator.device,dtype=vae.dtype) # ([24, 3, 1024, 768])
                cloth_values = vae.encode(cloth_values).latent_dist.sample() # ([24, 4, 128, 96])
                cloth_values = cloth_values * vae.config.scaling_factor

                cloth_filters = batch["cloth_filter"] # ([24, 1, 1024, 768])
                # resize the mask to latents shape as we concatenate the mask to the latents
                cloth_filter = torch.stack(
                    [
                        torch.nn.functional.interpolate(cloth_filters, size=(args.height // 8, args.width // 8))
                    ]
                )
                cloth_filter = cloth_filter.reshape(-1, 1, args.height // 8, args.width // 8)
                cloth_filter = cloth_filter.to(dtype = torch.float16)

                cloth_inputs = torch.cat([cloth_values, cloth_filter], dim=1) # ([24, 5, 128, 96])
                
                text_input_ids = tokenizer_one(
                    batch['caption_cloth'],
                    max_length=tokenizer_one.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                text_input_ids_2 = tokenizer_two(
                    batch['caption_cloth'],
                    max_length=tokenizer_two.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids


                # with torch.no_grad():
                encoder_output = text_encoder_one(text_input_ids.to(accelerator.device), output_hidden_states=True)
                text_embeds_cloth = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_two(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
                text_embeds_2_cloth = encoder_output_2.hidden_states[-2]
                text_embeds_cloth = torch.concat([text_embeds_cloth, text_embeds_2_cloth], dim=-1) # concat


                down,reference_features = UNet_Encoder(cloth_inputs.to(dtype=torch.float32),timesteps, text_embeds_cloth,return_dict=False)
                reference_features = list(reference_features)

                # prediction!!
                noise_pred = unet(latent_model_input.to(dtype=torch.float32), timesteps, encoder_hidden_states,added_cond_kwargs=unet_added_cond_kwargs,garment_features=reference_features).sample
                

                # setting loss target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")   


                # calculate loss
                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, 1.0)

                optimizer.step()
                optimizer.zero_grad()
                # Load scheduler, tokenizer and models.
                progress_bar.update(1)
                global_step += 1
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
            remaining_steps_in_epoch = num_update_steps_per_epoch - step - 1
            
            logs = {"step_loss": loss.detach().item(), "epoch": epoch, "remaining_steps_in_epoch": remaining_steps_in_epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break     
        # if global_step % (args.checkpointing_epoch * num_update_steps_per_epoch) == 0:
        if epoch % args.checkpointing_epoch == 0:
            if accelerator.is_main_process:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                # unwrapped_unet = accelerator.unwrap_model(
                #     unet, keep_fp32_wrapper=True
                # )
                
                unwrapped_unet_encoder = accelerator.unwrap_model(
                    UNet_Encoder, keep_fp32_wrapper=True
                )
                
                pipeline = TryonPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unet,
                    vae= vae,
                    scheduler=noise_scheduler,
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    text_encoder=text_encoder_one,
                    text_encoder_2=text_encoder_two,
                    image_encoder=image_encoder,
                    unet_encoder=unwrapped_unet_encoder,
                    torch_dtype=torch.float16,
                    add_watermarker=False,
                    safety_checker=None,
                )
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True) 
                pipeline.save_pretrained(save_path)
                del pipeline
                print("checkpoint save!")
            

if __name__ == "__main__":
    main()                    
