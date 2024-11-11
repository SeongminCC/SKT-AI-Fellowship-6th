import os
import re
import torch
import glob
from skimage import io
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from src.utils.util import get_fps, read_frames

from preprocess.focus_tunnel import PoseProcessingPipeline


class VitonHDDataset_front(Dataset):
    def __init__(self, base_dir, mode, transforms):  # base_dir : /home/vton/data/VITON-HD/
        self.base_dir = base_dir + mode + '/'
        self.transform = transforms
        self.mode = mode
        self.pipeline = PoseProcessingPipeline()
        
    def __len__(self):
        return len(glob.glob(self.base_dir + 'image/*.jpg'))

    def __getitem__(self, idx):
        img_path = sorted(glob.glob(os.path.join(self.base_dir, 'image', '*.jpg')))[idx]
        agnostic_path = img_path.replace("image","agnostic-v3.2")
        agn_mask_path = img_path.replace(".jpg","_mask.png").replace("image","agnostic-mask")
        pose_img_path = img_path.replace("image","dwpose")
        
        # Zoomed Image
        image_pil = Image.open(img_path).convert("RGB")
        ori_w, ori_h = image_pil.size
        image, tunnel_bbox, tunnel_info = self.pipeline.process_image(np.array(image_pil), expansion_ratio=1.46, prev_state=None)
        image = self.transform(Image.fromarray(image))
        
        # Zoomed Pose Image
        pose_img_pil = Image.open(pose_img_path).convert("RGB").resize(image_pil.size)
        cropped_pose_img = self.pipeline.crop_image(pose_img_pil, tunnel_bbox)
        pose_img = self.pipeline.pad_image_to_square(cropped_pose_img)
        pose_img = self.transform(Image.fromarray(pose_img))

        # Zoomed Agnostic
        agn_array = np.array(Image.open(agnostic_path).convert("RGB").resize(image_pil.size))
        cropped_agn_img = self.pipeline.crop_image(agn_array, tunnel_bbox)
        agnostic_img = self.pipeline.pad_image_to_square(cropped_agn_img)
        agnostic = self.transform(Image.fromarray(agnostic_img))
           
        # Zoomed Agnostic Mask
        agn_mask_array = np.array(Image.open(agn_mask_path).convert("RGB").resize(image_pil.size))
        cropped_agn_mask_img = self.pipeline.crop_image(agn_mask_array, tunnel_bbox)
        agnostic_mask_img = self.pipeline.pad_image_to_square(cropped_agn_mask_img)
        agn_mask = self.transform(Image.fromarray(agnostic_mask_img))
        
        # garment
        cloth_image_path = img_path.replace("image","cloth")
        cloth_image = self.transform(Image.open(cloth_image_path).convert("RGB"))

        # fil_cloth = self.transform(edge_filter(img_path=cloth_image_path, filter_name="sobel"))

        return {
            'images': image,
            'agnostic': agnostic,
            'agn_mask': agn_mask,
            'pose_img': pose_img,
            'cloth_image': cloth_image,
            # 'cloth_mask': cloth_mask,
            # 'fil_cloth' : fil_cloth,
        }
    
    
class ViViDDataset_front_image(Dataset):
    def __init__(self, base_dir, mode, transforms):  # base_dir : /home/vton/westchaevi/Tunnel/data/vivid_front_clip/
        self.base_dir = base_dir + mode + '/'
        self.transform = transforms
        self.mode = mode
        self.pipeline = PoseProcessingPipeline()

        
    def __len__(self):
        return len(glob.glob(self.base_dir + 'frames_videos/*.jpg'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx): # 인덱스가 tensor형태일 수 있는 것을 방지
              idx = idx.tolist()
                
        image_path = sorted(glob.glob(os.path.join(self.base_dir, 'frames_videos', '*.jpg')))[idx] 
        agnostic_path = image_path.replace("frames_videos","frames_agnostic").replace("front_","")
        agn_mask_path = image_path.replace("frames_videos","frames_agnostic_mask").replace("front_","")
        pose_img_path = image_path.replace("frames_videos","dwpose")
        
        # Zoomed Image
        image_pil = Image.open(image_path).convert("RGB")
        ori_w, ori_h = image_pil.size
        image, tunnel_bbox, tunnel_info = self.pipeline.process_image(np.array(image_pil), expansion_ratio=1.46, prev_state=None)
        image = self.transform(Image.fromarray(image))
        
        # Zoomed Pose Image
        pose_img_pil = Image.open(pose_img_path).convert("RGB").resize(image_pil.size)
        cropped_pose_img = self.pipeline.crop_image(pose_img_pil, tunnel_bbox)
        pose_img = self.pipeline.pad_image_to_square(cropped_pose_img)
        pose_img = self.transform(Image.fromarray(pose_img))

        # Zoomed Agnostic
        agn_array = np.array(Image.open(agnostic_path).convert("RGB").resize(image_pil.size))
        cropped_agn_img = self.pipeline.crop_image(agn_array, tunnel_bbox)
        agnostic_img = self.pipeline.pad_image_to_square(cropped_agn_img)
        agnostic = self.transform(Image.fromarray(agnostic_img))
           
        # Zoomed Agnostic Mask
        agn_mask_array = np.array(Image.open(agn_mask_path).convert("RGB").resize(image_pil.size))
        cropped_agn_mask_img = self.pipeline.crop_image(agn_mask_array, tunnel_bbox)
        agnostic_mask_img = self.pipeline.pad_image_to_square(cropped_agn_mask_img)
        agn_mask = self.transform(Image.fromarray(agnostic_mask_img))

        # Assuming cloth images are named similarly to videos
        cloth_image_path = re.sub(r'frames_videos/(\d+)_detail_\d+_front.mp4', r'cloth/\1_in_xl.jpg', image_path)
        cloth_image = self.transform(Image.open(cloth_image_path).convert("RGB"))
        # cloth_mask = self.transform(Image.open(cloth_mask_path).convert("RGB"))
        # fil_cloth = self.transform(edge_filter(img_path=cloth_image_path, filter_name="sobel"))

        return {
            'images': image,
            'agnostic': agnostic,
            'agn_mask': agn_mask,
            'pose_img': pose_img,
            'cloth_image': cloth_image,
        }
    

class ViViDDataset_front_video(Dataset):
    def __init__(self, base_dir, mode, transforms):  # base_dir : /home/vton/westchaevi/Tunnel/data/vivid_front_clip/videos
        self.base_dir = base_dir + mode + '/'
        self.transform = transforms
        self.mode = mode

        
    def __len__(self):
        return len(glob.glob(self.base_dir + 'videos/*_front.mp4'))

    def __getitem__(self, idx):
        video_path = sorted(glob.glob(os.path.join(self.base_dir, 'videos', '*_front.mp4')))[idx]

        video_tensor_list=[]
        video_images = read_frames(video_path)
        
        for vid_image_pil in video_images:
            video_tensor_list.append(self.transform(vid_image_pil))
            
        video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
        # video_tensor = video_tensor.transpose(0, 1)           # (c, f, h, w)
        
        agnostic_list=[]
        agnostic_images=read_frames(agnostic_path)
        for agnostic_image_pil in agnostic_images:
            agnostic_list.append(self.transform(agnostic_image_pil))

        agn_mask_list=[]
        agn_mask_images=read_frames(agn_mask_path)
        for agn_mask_image_pil in agn_mask_images:
            agn_mask_list.append(self.transform(agn_mask_image_pil))

        pose_list=[]
        pose_images=read_frames(densepose_path)
        for pose_image_pil in pose_images:
            pose_list.append(self.transform(pose_image_pil))
        

        # Assuming cloth images are named similarly to videos
        cloth_image_path = re.sub(r'videos/(\d+)_detail_\d+_front.mp4', r'cloth/\1_in_xl.jpg', video_path)
        # Load cloth image and mask
        cloth_image = self.transform(Image.open(cloth_image_path).convert("RGB"))
        
        return {
            'video_frames': video_tensor,
            'agnostic_frames': agnostic_list,
            'agn_mask_frames': agn_mask_list,
            'openpose_frames': pose_list,
            'cloth_image': cloth_image,
        }
    


# Define Dataset
class LouisVTONDataset_front(Dataset):
    def __init__(self, base_dir, mode, transforms):  # base_dir : /home/vton/westchaevi/ViViD/data_cleve/
        self.base_dir = base_dir + mode + '/'
        self.transform = transforms
        self.mode = mode
        
    def __len__(self):
        return len(glob.glob(self.base_dir + 'videos/*_front.mp4'))

    def __getitem__(self, idx):
        video_path = sorted(glob.glob(os.path.join(self.base_dir, 'videos', '*_front.mp4')))[idx]
        agnostic_path = video_path.replace("_front.mp4",".mp4").replace("videos","agnostic")
        agn_mask_path = video_path.replace("_front.mp4",".mp4").replace("videos","agnostic_mask")
        densepose_path = video_path.replace("_front.mp4",".mp4").replace("videos","densepose")

        video_tensor_list=[]
        video_images = read_frames(video_path)
        
        for vid_image_pil in video_images:
            video_tensor_list.append(self.transform(vid_image_pil))
            
        video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
        # video_tensor = video_tensor.transpose(0, 1)           # (c, f, h, w)
        
        agnostic_list=[]
        agnostic_images=read_frames(agnostic_path)
        for agnostic_image_pil in agnostic_images:
            agnostic_list.append(self.transform(agnostic_image_pil))

        agn_mask_list=[]
        agn_mask_images=read_frames(agn_mask_path)
        for agn_mask_image_pil in agn_mask_images:
            agn_mask_list.append(self.transform(agn_mask_image_pil))

        pose_list=[]
        pose_images=read_frames(densepose_path)
        for pose_image_pil in pose_images:
            pose_list.append(self.transform(pose_image_pil))
        

        # Assuming cloth images are named similarly to videos
        cloth_image_path = re.sub(r'agnostic/(\d+)_\d+.mp4', r'cloth/\1.jpg', agnostic_path)
        cloth_mask_path = cloth_image_path.replace('cloth', 'cloth_mask')


        # Load cloth image and mask
        cloth_image = self.transform(Image.open(cloth_image_path).convert("RGB"))
        cloth_mask = self.transform(Image.open(cloth_mask_path).convert("RGB"))
        fil_cloth = self.transform(edge_filter(img_path=cloth_image_path, filter_name="sobel"))

        return {
            'video_frames': video_tensor,
            'agnostic_frames': agnostic_list,
            'agn_mask_frames': agn_mask_list,
            'densepose_frames': pose_list,
            'cloth_image': cloth_image,
            'cloth_mask': cloth_mask,
            'fil_cloth' : fil_cloth,
        }
    
    
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