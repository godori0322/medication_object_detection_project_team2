# src/config.py

from pathlib import Path
import argparse

# --- 기본 경로 설정 ---
# 이 파일(config.py)의 부모 디렉토리(src)의 부모 디렉토리(deep-learning-project)를 기준 경로로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- 데이터 경로 ---
DATA_DIR = BASE_DIR / "data/ai03-level1-project"
TRAIN_IMAGE_DIR = DATA_DIR / "train_images"
TEST_IMAGE_DIR = DATA_DIR / "test_images"
TRAIN_ANNOTATION_DIR = DATA_DIR / "train_annotations" 

# --- 결과물 경로 ---
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# --- 폴더 생성 ---
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --- 학습 하이퍼파라미터 ---
DEVICE = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 44199
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005

# --- 평가 설정 ---
CONFIDENCE_THRESHOLD = 0.5
MODEL_CHECKPOINT = CHECKPOINT_DIR / "pill_detector_best.pth"


# argparse를 이용한 유동적인 하이퍼파라미터 조정
def get_config():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    return args