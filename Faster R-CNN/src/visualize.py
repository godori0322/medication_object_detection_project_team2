# src/visualize.py

import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont # ImageDraw, ImageFont 임포트 추가

from . import config
from .dataset import PillDataset
from .model import get_detection_model
from .utils import get_class_mapping

# --- 폰트 경로 지정 (가장 일반적인 우분투 경로) ---
# 이 경로에 폰트가 없으면, 설치된 경로로 수정해야 할 수 있습니다.
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

def visualize_prediction(image_path, model, class_mapping):
    """ 모델의 예측 결과를 시각화 (한글 깨짐 해결 버전) """
    
    # 1. 이미지 로드 (Pillow 사용)
    image_pil = Image.open(image_path).convert("RGB")
    
    # 2. 모델 추론을 위한 텐서 변환
    image_tensor = torch.as_tensor(np.array(image_pil), dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(config.DEVICE)

    # 3. 모델 예측 실행
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])

    # 4. 시각화를 위해 Pillow Draw 객체 및 폰트 로드
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, 15) # 텍스트용 폰트
        title_font = ImageFont.truetype(FONT_PATH, 20) # 제목용 폰트
    except IOError:
        print(f"'{FONT_PATH}'에서 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # 5. 예측 결과 그리기
    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > config.CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = map(int, box)
            
            # 약제 이름 조회
            class_id_str = str(label.item())
            drug_name = class_mapping.get(class_id_str, f"ID: {class_id_str}")
            text = f"{drug_name} ({score:.2f})"
            
            # 바운딩 박스 그리기 (파란색)
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 0, 255), width=3)
            
            # 텍스트 그리기 (Pillow 사용)
            draw.text((x_min, y_min - 20), text, font=font, fill=(255, 0, 0))

    # 6. Matplotlib으로 최종 이미지 출력
    plt.figure(figsize=(12, 12))
    plt.imshow(image_pil)
    plt.title("모델 예측 결과", fontproperties={'fname': FONT_PATH, 'size': 16}) # 제목 폰트 직접 지정
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print("테스트 이미지에 대한 모델 예측을 시각화합니다.")
    
    class_mapping = get_class_mapping()
    device = torch.device(config.DEVICE)
    model = get_detection_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
        
        test_images = list(Path(config.TEST_IMAGE_DIR).glob("*.png"))
        if test_images:
            random_image_path = random.choice(test_images)
            visualize_prediction(random_image_path, model, class_mapping)
        else:
            print("테스트 이미지를 찾을 수 없습니다.")

    except FileNotFoundError:
        print(f"오류: 모델 체크포인트 파일을 찾을 수 없습니다. 경로: {config.MODEL_CHECKPOINT}")
        print("모델을 먼저 학습시켜주세요.")