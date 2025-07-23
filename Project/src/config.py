# src/config.py

## 주의! argparse는 기본적으로 CLI 실행을 가정하기 때문에, 
## Jupyter Notebook이나 Colab에서 error: unrecognized arguments 발생 가능
## import sys
## sys.argv = ['']  # argparse 충돌 방지용

## from config import get_config
## cfg = get_config()
## 위 코드를 통해 해결하세요. 

from pathlib import Path
import torch
import torch.optim as optim
import os
import sys
import argparse

BATCH_SIZE = 16
NUM_WORKERS = 0

# --- 기본 경로 설정 ---
# 이 파일(config.py)의 부모 디렉토리(src)의 부모 디렉토리(Project)를 기준 경로로 설정
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Colab에서 __file__이 없는 경우, 수동 경로 지정
    BASE_DIR = Path(os.getcwd()).resolve()  # 현재 작업 디렉토리

# --- 데이터 경로 ---
DATA_DIR = BASE_DIR / "data" / "ai03-level1-project"
TRAIN_IMAGE_DIR = DATA_DIR / "train_images"
TEST_IMAGE_DIR = DATA_DIR / "test_images"
TRAIN_ANNOTATION_DIR = DATA_DIR / "train_annotations"

CSV_TRAIN_DATA_DIR = BASE_DIR / "data_csv"
FILLTER_CSV_TRAIN_DATA_DIR = CSV_TRAIN_DATA_DIR / "model_train_data_csv"

# --- 결과물 경로 ---
OUTPUT_DIR = BASE_DIR / "Project" / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# --- 평가 설정 ---
MODEL_CHECKPOINT = CHECKPOINT_DIR / "pill_detector_best.pth"

# Windows, Mac 이용 여부에 따른 device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# config에서 optimizer 관리하여 실험 용이하게
def get_optimizer(model, cfg):
    if cfg.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
         return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

# argparse를 이용한 유동적인 하이퍼파라미터 조정
def get_config():
    if any(env in sys.modules for env in ['google.colab', 'ipykernel']):
        sys.argv = ['']  # Jupyter/Colab에서 argparse 충돌 방지

    parser = argparse.ArgumentParser(description="Training configuration")

    default_device = get_device()

    parser.add_argument('--device', type=str, default=default_device, help='Device to use (cuda, mps or cpu)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_classes', type=int, default=44199, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimzer') # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--momentum', type=float, default=0.01, help='Momentum')

    args = parser.parse_args()
    
    args.base_dir = BASE_DIR
    args.data_dir = DATA_DIR
    args.train_image_dir = TRAIN_IMAGE_DIR
    args.test_image_dir = TEST_IMAGE_DIR
    args.annotation_dir = TRAIN_ANNOTATION_DIR
    args.output_dir = OUTPUT_DIR
    args.checkpoint_dir = CHECKPOINT_DIR
    args.model_checkpoint = MODEL_CHECKPOINT
    args.log_dir = LOG_DIR

    return args