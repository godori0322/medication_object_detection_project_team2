# 💊 경구약제 이미지 객체 검출 프로젝트

Python 3.11.9 환경에서 실행되며, 모듈별로 기능을 분리하여 **응집도는 최대화**, **결합도는 최소화**한 객체 검출(Object Detection) 프로젝트입니다.  
데이터 전처리 및 모델 학습 파이프라인 활용하여 경구 복용 약제(알약)를 이미지에서 정확히 탐지하는 것을 목표로 합니다.

---

## 🧱 프로젝트 구조 및 모듈 설명

project/

│

├── src/                   # 주요 소스코드

│   └── main.py            # CLI에서 one line command로 main 호출 시 데이터 전처리, 모델 학습 및 결과 저장까지 일괄 처리되도록 설계

│   └── models/

│       ├── yolo_v5.py

│       ├── yolo_v8.py

│       ├── yolo_v11.py

│       ├── faster_rcnn.py

│       └── ssd.py

│   └── train/

│       ├── pytorch_train.py

│       └── yolo_train.py

│   └── utils/

│   └── config.py

│   └── dataset.py

│   └── dataloader.py

│   └── test.py

│   └── yolo_test.py
