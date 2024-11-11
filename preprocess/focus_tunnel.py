import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque
from PIL import Image
from src.dwpose.wholebody import Wholebody


class PoseProcessingPipeline:
    def __init__(self, kalman_window_size=5, lowpass_window_size=5, cuda=0):
        # PoseMap 추출을 위한 Detector 초기화
        
        self.detector = Wholebody(device='cpu')  # GPU 사용 (필요 시 'cpu'로 변경)

        # Kalman 필터 설정
        self.kf = None  # Kalman 필터 초기 상태 (None)
        self.is_new_video = True  # 새로운 비디오인지 여부를 추적

        # Low-pass 필터 설정
        self.low_pass_filter = LowPassFilter(window_size=lowpass_window_size)

    def extract_pose(self, image):
        """
        PoseMap 추출 함수.
        이미지를 입력받아 포즈 좌표를 추출하고 반환.
        """
        keypoints, scores = self.detector(image)
        return keypoints
    
    
#     def generate_minimum_bbox(self, keypoints):
#         """
#         상반신에 해당하는 keypoints를 사용하여 최소 경계 상자를 생성.
#         머리 부분이 있을 경우 min_y를 머리로 설정, 없을 경우 어깨를 기준으로 설정.
#         keypoints: np.ndarray of shape (1, num_keypoints, 2)
#         """
#         # 상반신 키포인트 인덱스 (목, 어깨, 팔꿈치, 손목, 머리)
#         upper_body_indices = [1, 2, 3, 4, 5, 6, 7, 11]
#         head_index = 32  # 머리의 인덱스 (보통 0이 머리로 사용됨)

#         keypoints = keypoints[0]

#         # 상반신에 해당하는 키포인트들만 추출
#         upper_body_keypoints = keypoints[upper_body_indices]
#         valid_keypoints = upper_body_keypoints[upper_body_keypoints[:, 0] != -1]

#         if valid_keypoints.size == 0:
#             raise ValueError("유효한 상반신 좌표가 없습니다.")

#         # 머리 좌표가 있는지 확인
#         head_keypoint = keypoints[head_index]
#         if head_keypoint[0] != -1:  # 머리 좌표가 유효하다면
#             min_y = head_keypoint[1]  # 머리 좌표의 y값을 min_y로 설정
#         else:
#             min_y = np.min(valid_keypoints[:, 1])  # 어깨 기준 min_y

#         # 최소 경계 상자 생성
#         min_x = np.min(valid_keypoints[:, 0])
#         max_x = np.max(valid_keypoints[:, 0])
#         max_y = np.max(valid_keypoints[:, 1])

        return [min_x, min_y, max_x, max_y]

    def generate_minimum_bbox(self, keypoints):
        """
        상반신에 해당하는 keypoints를 사용하여 최소 경계 상자를 생성.
        머리 부분이 있을 경우 min_y를 머리로 설정, 없을 경우 어깨를 기준으로 설정.
        keypoints: np.ndarray of shape (1, num_keypoints, 2)
        """
        # 상반신 키포인트 인덱스 (목, 어깨, 팔꿈치, 손목, 머리)
        upper_body_indices = [1, 2, 3, 4, 5, 6, 7]
        head_index = 32  # 머리의 인덱스 (보통 0이 머리로 사용됨)

        keypoints = keypoints[0]

        # 상반신에 해당하는 키포인트들만 추출
        upper_body_keypoints = keypoints[upper_body_indices]
        valid_keypoints = upper_body_keypoints[upper_body_keypoints[:, 0] != -1]

        if valid_keypoints.size == 0:
            raise ValueError("유효한 상반신 좌표가 없습니다.")

        # # 머리 좌표가 있는지 확인
        # head_keypoint = keypoints[head_index]
        # if head_keypoint[0] != -1:  # 머리 좌표가 유효하다면
        #     min_y = head_keypoint[1]  # 머리 좌표의 y값을 min_y로 설정
        # else:
        #     min_y = np.min(valid_keypoints[:, 1])  # 어깨 기준 min_y

        # 최소 경계 상자 생성
        min_y = np.min(keypoints[:, 1])
        min_x = np.min(valid_keypoints[:, 0])
        max_x = np.max(valid_keypoints[:, 0])
        max_y = np.max(valid_keypoints[:, 1])

        return [min_x, min_y, max_x, max_y]


    def expand_bbox(self, image, bbox, expansion_ratio=1.5):
        """
        확장된 bounding box 생성.
        expansion_ratio: 확장 비율
        """
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        min_x -= (width * (expansion_ratio - 1)) / 2
        max_x += (width * (expansion_ratio - 1)) / 2
        min_y -= (height * (expansion_ratio - 1)) / 2
        max_y += (height * (expansion_ratio - 1)) / 2
        
        min_x_ = max(0, min_x)  
        min_y_ = max(0, min_y)
        max_x_ = min(image.shape[1], max_x)
        max_y_ = min(image.shape[0], max_y)
        

        return [min_x_, min_y_, max_x_, max_y_]
    
    def shrink_bbox(self, image, bbox, shrink_ratio=1.5):
        """
        축소된 bounding box 생성.
        shrink_ratio: 축소 비율
        """
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        # width와 height를 shrink_ratio에 맞춰 축소
        min_x += (width * (shrink_ratio - 1)) / 2
        max_x -= (width * (shrink_ratio - 1)) / 2
        # min_y += (height * (shrink_ratio - 1)) / 2  # 필요에 따라 사용 가능
        max_y -= (height * (shrink_ratio - 1)) / 2

        # 이미지 경계 내로 제한
        min_x_ = max(0, min_x)
        min_y_ = max(0, min_y)
        max_x_ = min(image.shape[1], max_x)
        max_y_ = min(image.shape[0], max_y)

        return int(min_x_), int(min_y_), int(max_x_), int(max_y_)

    
    def initialize_kalman_filter(self):
        """
        새로운 비디오가 시작될 때 Kalman 필터 초기화.
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=4)
        self.kf.F = np.eye(4)  # 상태 전이 행렬
        self.kf.H = np.eye(4)  # 측정 행렬
        self.kf.P *= 1000.  # 초기 불확실성
        self.kf.R = np.eye(4) * 5  # 측정 노이즈
        self.kf.Q = np.eye(4) * 0.1  # 프로세스 노이즈
        
    def start_new_video(self):
        """
        새로운 비디오가 시작될 때 호출하여 Kalman 필터 초기화.
        """
        self.is_new_video = True

    def apply_kalman_filter(self, bbox, prev_state=None):
        """
        Kalman 필터 적용. 첫 번째 프레임일 경우 필터 적용 없이 bbox 그대로 반환.
        """
        # prev_state가 None이면 첫 번째 프레임으로 간주하고, 필터를 적용하지 않음
        if self.is_new_video or prev_state is None:
            # 첫 번째 프레임이거나 새로운 비디오일 경우 Kalman 필터 초기화
            self.initialize_kalman_filter()
            self.is_new_video = False  # 새로운 비디오 처리 중 상태로 변경
            return np.array(bbox)  # bbox 그대로 반환

        # Kalman 필터 상태 업데이트
        self.kf.x = prev_state
        self.kf.predict()
        self.kf.update(np.array(bbox))
        kalman_bbox = self.kf.x

        # Low-pass 필터 적용
        smoothed_bbox = self.low_pass_filter.apply(kalman_bbox)

        return smoothed_bbox


    def crop_image(self, image, bbox):
        """
        이미지에서 bounding box에 맞게 잘라내기.
        """
    
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        min_x, min_y, max_x, max_y = [int(coord) for coord in bbox]
        min_x_ = max(0, min_x)  
        min_y_ = max(0, min_y)
        max_x_ = min(image.shape[1], max_x)
        max_y_ = min(image.shape[0], max_y)

        cropped_img = image[min_y_:max_y_, min_x_:max_x_]
        return cropped_img
    

    def pad_image_to_square(self, image, target_aspect_ratio=(3,4)):
        # Step 1: 입력 이미지의 가로, 세로 크기 계산
        height, width = image.shape[:2]

        # Step 2: 현재 이미지의 비율과 목표 비율 계산
        current_aspect_ratio = width / height
        target_width, target_height = target_aspect_ratio
        target_ratio = target_width / target_height  # 4:3 비율 계산

        # Step 3: 이미지가 더 넓으면 (현재 비율이 4:3보다 크면)
        if current_aspect_ratio > target_ratio:
            # 가로를 기준으로 비율에 맞게 세로에 패딩 추가
            new_height = int(width / target_ratio)
            pad_height = (new_height - height) // 2
            padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Step 4: 이미지가 더 길면 (현재 비율이 4:3보다 작으면)
        elif current_aspect_ratio < target_ratio:
            # 세로를 기준으로 비율에 맞게 가로에 패딩 추가
            new_width = int(height * target_ratio)
            pad_width = (new_width - width) // 2
            padded_image = cv2.copyMakeBorder(image, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Step 5: 이미 비율이 맞다면 패딩 없이 원본 반환
        else:
            padded_image = image

        return padded_image
    

    def process_image(self, image, expansion_ratio=1.62, prev_state=None):
        """
        전체 파이프라인 실행:
        1. PoseMap 추출
        2. 최소 경계 상자 생성
        3. 확장 및 필터링 (Kalman + Low-pass)
        4. 후처리 (Crop, Padding, Resize)
        """
        
        # 1. PoseMap 추출
        keypoints = self.extract_pose(image)

        # 2. 최소 경계 상자 생성
        bbox = self.generate_minimum_bbox(keypoints)

        # 3. Bounding Box 확장 및 Kalman + Low-pass 필터 적용
        expanded_bbox = self.expand_bbox(image, bbox, expansion_ratio)
        smoothed_bbox = self.apply_kalman_filter(expanded_bbox, prev_state)

        # 4. 후처리
        cropped_img = self.crop_image(image, smoothed_bbox)
        padded_img = self.pad_image_to_square(cropped_img)
        final_img = cv2.resize(padded_img, (768, 1024), interpolation=cv2.INTER_LINEAR)
        
        # tunnel info
        min_x, min_y, max_x, max_y = smoothed_bbox
        
        tunnel_center = ((min_x + max_x) // 2 , (min_y + max_y) // 2)

        tunnel_width = max_x - min_x
        tunnel_height = max_y - min_y
        tunnel_size = (int(tunnel_width), int(tunnel_height))
        
        tunnel_info = {}
        tunnel_info['ori_res'] = (image.shape[1], image.shape[0])
        tunnel_info['tunnel_center'] = tunnel_center
        tunnel_info['tunnel_size'] = tunnel_size

        return final_img, smoothed_bbox, tunnel_info
    
    



class LowPassFilter:
    def __init__(self, window_size=5):
        """
        Low-pass 필터 초기화 (이동 평균 필터)
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def apply(self, bbox):
        """
        Low-pass 필터 적용 (이동 평균)
        """
        self.history.append(bbox)
        avg_bbox = np.mean(self.history, axis=0)
        return avg_bbox
