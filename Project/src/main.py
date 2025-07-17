# src/main.py

from . import train
from .config import get_config

def main():
    config = get_config()

    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")

    # config를 사용한 실행 코드 예시
    # python train.py --epochs 30 --batch_size 32 --lr 0.0005 --device cpu
    
    print("Start Training")
    train.main()

if __name__ == "__main__":
    main()