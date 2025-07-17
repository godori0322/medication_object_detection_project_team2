import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

def save_loss_curve(train_losses, val_losses, num_epochs, save_path):
    """손실 곡선을 저장하는 함수"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', linestyle='--', label='Validation Loss')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.xticks(range(1, num_epochs + 1))
    plt.savefig(save_path)
    print(f"📈 Loss curve saved to {save_path}")

def visualize_prediction(image_path, model, class_mapping, config):
    """모델의 예측 결과를 시각화"""
    
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    
    # Convert to tensor for model
    image_tensor = torch.as_tensor(np.array(image_pil), dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(config.DEVICE)

    # Model prediction
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])

    # Setup drawing
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, 15)
    except IOError:
        print(f"Font not found at '{FONT_PATH}'. Using default font.")
        font = ImageFont.load_default()

    # Draw predictions
    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score > config.CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = map(int, box)
            
            class_id_str = str(label.item())
            drug_name = class_mapping.get(class_id_str, f"ID: {class_id_str}")
            text = f"{drug_name} ({score:.2f})"
            
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 0, 255), width=3)
            draw.text((x_min, y_min - 20), text, font=font, fill=(255, 0, 0))

    # Display result
    plt.figure(figsize=(12, 12))
    plt.imshow(image_pil)
    plt.title("Model Prediction Results")
    plt.axis('off')
    plt.show()

def visualize_predictions(model, class_mapping, config):
    """시각화 실행 함수"""
    print("🎨 테스트 이미지에 대한 모델 예측을 시각화합니다.")
    
    test_images = list(Path(config.TEST_IMAGE_DIR).glob("*.png"))
    if test_images:
        random_image_path = random.choice(test_images)
        visualize_prediction(random_image_path, model, class_mapping, config)
    else:
        print("테스트 이미지를 찾을 수 없습니다.")