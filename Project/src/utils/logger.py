# src/utils/logger.py

import os
import csv
from pathlib import Path
from datetime import datetime

# 결과 담을 experiment 폴더 생성
def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

class Logger:
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_hyperparameters_csv(self, params: dict, filename="hyperparameters.csv"):
        """
        하이퍼파라미터 dict를 CSV 파일로 저장

        Args:
            params (dict): 저장할 하이퍼파라미터 딕셔너리
            filename (str): 저장할 파일명 (기본 "hyperparameters.csv")
        """
        save_path = self.save_dir / filename
        with open(save_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 (키)
            writer.writerow(params.keys())
            # 값 (value)
            writer.writerow(params.values())
        print(f"Hyperparameters saved to: {save_path}")