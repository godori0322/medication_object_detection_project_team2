# src/utils/logger.py

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str = "pill-detection", save_dir: str = "experiments", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 중복 방지
    if logger.hasHandlers():
        return logger

    # 포맷터 설정
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 파일 핸들러 (선택)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger