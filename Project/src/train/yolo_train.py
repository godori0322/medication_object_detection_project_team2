import yaml
from pathlib import Path
from ultralytics import YOLO

def train_yolo(model: YOLO, cfg) -> YOLO:
    """
    Train (or tune → train) a YOLO model according to cfg settings.

    Modes:
      1) cfg.tune=True: hyperparameter tuning → load best hyp → final train
      2) cfg.hyp_path set: load provided hyp file → train
      3) 기본 모드: cfg.lr 등 직접 지정 → train

    Args:
        model: ultralytics YOLO 모델 인스턴스
        cfg:   설정 객체, 필수 속성:
               - data_dir: str/Path, data.yaml 위치
               - output_dir: str 저장 디렉터리
               - device: 'cpu' or 'cuda'
               - batch_size: int
               - optimizer: 'AdamW' or 'SGD'
               - num_epochs: int
               - tune: bool
               - tune_epochs, iterations: 튠시
               - hyp_path: Optional[str]
               - lr, lrf, momentum, weight_decay: 기본 학습시
    Returns:
        학습 완료된 YOLO 모델
    """
    data_yaml = Path(cfg.data_dir) / "data.yaml"     # 데이터 구성 파일
    out_dir = Path(cfg.output_dir)                    # 출력 경로
    opt_lower = cfg.optimizer.lower()                 # optimizer 구분용

    # 1) 튜닝 모드
    if getattr(cfg, "tune", False):
        tune_kwargs, search_space = _build_tune_config(opt_lower, cfg, data_yaml, out_dir)
        model.tune(**tune_kwargs, space=search_space)
        # 최신 튠 결과에서 최적 파라미터 로드
        latest = sorted(
            [p for p in out_dir.iterdir() if p.name.startswith("yolo_tune")],
            key=lambda p: p.stat().st_mtime
        )[-1]
        with open(latest / "best_hyperparameters.yaml", encoding="utf-8") as f:
            hyperparams = yaml.safe_load(f)

    # 2) 외부 hyp 파일 모드
    elif getattr(cfg, "hyp_path", None):
        hyp_file = Path(cfg.hyp_path)
        if not hyp_file.exists():
            raise FileNotFoundError(f"HYP file not found: {hyp_file}")
        with open(hyp_file, encoding="utf-8") as f:
            hyperparams = yaml.safe_load(f)

    # 3) 기본 학습 모드
    else:
        hyperparams = {
            "lr0":          cfg.lr,            # 초기 학습률
            "lrf":          cfg.lrf,           # 최종 학습률 비율
            "weight_decay": cfg.weight_decay,  # 가중치 감쇠
            "momentum":     cfg.momentum,      # 옵티마이저 모멘텀
        }

    # 공통 train() 인자
    common_args = {
        # 데이터 및 반복
        "data":             str(data_yaml),    # data.yaml 경로
        "epochs":           cfg.num_epochs,    # 학습 epoch 수
        "imgsz":            640,               # 입력 이미지 크기
        "batch":            cfg.batch_size,    # 배치 크기
        "device":           cfg.device,        # 연산 장치
        # 최적화
        "optimizer":        cfg.optimizer,     # AdamW 또는 SGD
        # 학습 스케줄
        "rect":             False,             # 고정 비율 배치
        "cos_lr":           True,              # cosine LR 스케줄
        "patience":         20,                # early stop patience
        # 출력 및 저장
        "plots":            True,              # 학습 곡선 저장
        "save":             True,              # 모델 체크포인트 저장
        "project":          str(out_dir),      # 프로젝트 디렉터리
        "name":             "yolo_experiment", # 실험 이름
        # 증강
        # "augment":          True,              # 증강 사용여부
        # "auto_augment":     "auto_augment",    # 증강 최적화
        # "copy_paste_mode":  "mixup",           # mixup 모드
        # 기타
        # "close_mosaic":     20,                # mosaic 종료 epoch
        **hyperparams,                         # 튜닝/파일/기본 파라미터
    }

    model.train(**common_args)
    print("YOLO 학습 완료")
    return model


def _build_tune_config(opt: str, cfg, data_yaml: Path, out_dir: Path):
    """
    cfg.tune=True 시 tune() 호출 인자와 search_space 반환.
    """
    # 공통 튠 인자
    base_kwargs = {
        "data":         str(data_yaml),      # data.yaml
        "device":       cfg.device,          # 연산 장치
        "batch":        cfg.batch_size,      # 배치 크기
        "imgsz":        320,                 # 튠 이미지 크기
        "patience":     5,                   # early stop patience
        "rect":         False,               # 고정 비율 배치
        "val":          True,                # 검증 실행
        "plots":        False,               # 튠 플롯 저장
        "save":         False,               # 튠 모델 저장
        "cos_lr":       False,               # cosine LR 스케줄
        "half":         True,                # half precision
        "cache":        "ram",               # 캐시 옵션
        "project":      str(out_dir),        # 프로젝트
        "name":         "yolo_tune",         # 실험 이름
        "augment":      False,               # 증강 사용여부
        "auto_augment": None,                # 증강 최적화
        "epochs":       cfg.tune_epochs,     # 튠 epoch
        "iterations":   cfg.iterations,      # 튠 iterations
    }

    """
    ### Optimizer별 차이를 간단 정리 (주석)
    - **AdamW**
        * 적응형 스케일링(1/√v_t) → 같은 절대 LR라도 실제 update step이 크다.
        * decoupled weight‑decay라 규제항이 LR과 분리 ⇒ decay 값을 조금 크게 잡아도 안정.
        * β₂까지 포함된 모멘텀 덕분에 warm‑up 기간이 짧아도 초기 진동을 제어.
    - **SGD**
        * 고정 step + 1차 모멘텀만 존재 ⇒ 더 큰 초기 LR, 더 긴 warm‑up으로 관성 확보.
        * weight‑decay가 LR에 곱붙어 적용되므로 값이 크면 발산 위험 → 좁은 범위.
        * 모멘텀(0.85~0.98)을 폭넓게 튠해 수렴·진동 균형 맞춤.
    - **Loss weights**
        * AdamW는 grad norm이 커서 box/cls/dfl 가중치를 크게 잡아도 안정.
        * SGD는 큰 step 때문에 과한 가중치가 발산을 유발 → 다소 낮은 범위.

    ### 색상/기하·혼합 증강은 튠 단계에선 0 고정
        * 학습 단계에서 별도 조정할 때 사용.
        * 아래 ‘○’ 표기 뒤에 관례적 탐색 폭과 간단 이유를 함께 주석 처리.
    """
    if opt == "adamw":
        base_kwargs["optimizer"] = "AdamW"
        search_space = {
            # ---------- LR & 스케줄 ----------
            "lr0": (2e-4, 1e-3),   # 초기 LR ↓ : 적응형 scaling 덕분에 작은 값으로도 충분
            "lrf": (0.10, 0.45),   # 최종 LR 비율 ↑ : 낮은 lr0에서 decaying 효과 확보
            # ---------- 정규화 & 워밍업 ----------
            "weight_decay": (2e-4, 8e-4),  # decoupled L2 규제 → 0.0005 주변 ±60 %
            "warmup_epochs": (0.3, 1.5),   # 적응형 옵티마이저라 warm‑up 짧게
            "warmup_bias_lr": (0.03, 0.20),# bias 파라미터 variance 보정용
            # ---------- 손실 가중치 ----------
            "box": (6.0, 10.0),    # 객체 갯수 적은 데이터에 맞춰 기본값(7.5)±30 %
            "cls": (0.40, 1.60),
            "dfl": (1.0, 2.4),
            # ---------- 증강(0 고정) ----------
            "hsv_h": (0.00, 0.00),   # ○ 0~0.015 : ±5° Hue 변화(가짜색 최소화)
            "hsv_s": (0.00, 0.00),   # ○ 0~0.70 : 탈색~쨍한 범위
            "hsv_v": (0.00, 0.00),   # ○ 0~0.40 : 밝기 ±40 %, 실조도 한계
            "degrees": (0.0, 0.0),   # ○ 0~10°  : bbox 회전 오차 억제선
            "translate": (0.00, 0.00), # ○ 0~0.10 : 10 % 이동까지 framing 다양화
            "scale": (0.00, 0.00),   # ○ 0.50~1.50 : 해상도·픽셀 손실 최소
            "shear": (0.0, 0.0),     # ○ 0~2°   : 카메라 tilt 시뮬
            "perspective": (0.0, 0.0000), # ○ 0~0.001 : edge‑case 보강
            "flipud": (0.0, 0.00),   # ○ 0~0.10 : 상하반전은 드뭄
            "fliplr": (0.0, 0.00),   # ○ 0~0.50 : 좌우반전은 최대 50 %
            "mosaic": (0.00, 0.00),  # ○ 0~1.00 : 대표 혼합 증강
            "mixup": (0.00, 0.00),   # ○ 0~0.30 : 30 % ↑는 bbox 혼란
            "copy_paste": (0.00, 0.00), # ○ 0~0.30 : 객체밀도↑ 시만
            "cutmix": (0.00, 0.00),  # ○ 0~0.30 : label noise ↑ 주의
            "bgr": (0.00, 0.00),     # ○ 0 or 1  : 채널스왑 on/off
        }

    elif opt == "sgd":
        base_kwargs["optimizer"] = "SGD"
        search_space = {
            # ---------- LR & 스케줄 ----------
            "lr0": (3e-3, 3e-2),   # 고정 step이므로 큰 LR 필요
            "lrf": (0.05, 0.20),   # 종단 LR 비율 낮게 → 과도한 decaying 방지
            "momentum": (0.85, 0.98),  # 인공 관성 조절 핵심
            # ---------- 정규화 & 워밍업 ----------
            "weight_decay": (4e-5, 6e-4),  # LR에 곱붙어 적용 → 값이 클수록 불안
            "warmup_epochs": (0.5, 2.5),   # 큰 LR 때문에 좀 더 길게
            "warmup_momentum": (0.80, 0.95),
            "warmup_bias_lr": (0.05, 0.25),
            # ---------- 손실 가중치 ----------
            "box": (5.0, 8.0),    # 과한 가중치 시 발산 위험 → 범위 ↓
            "cls": (0.30, 1.20),
            "dfl": (1.0, 2.0),
            # ---------- 증강(0 고정) ----------
            "hsv_h": (0.00, 0.00),   # ○ 0~0.015 : ±5° Hue 변화(가짜색 최소화)
            "hsv_s": (0.00, 0.00),   # ○ 0~0.70 : 탈색~쨍한 범위
            "hsv_v": (0.00, 0.00),   # ○ 0~0.40 : 밝기 ±40 %, 실조도 한계
            "degrees": (0.0, 0.0),   # ○ 0~10°  : bbox 회전 오차 억제선
            "translate": (0.00, 0.00), # ○ 0~0.10 : 10 % 이동까지 framing 다양화
            "scale": (0.00, 0.00),   # ○ 0.50~1.50 : 해상도·픽셀 손실 최소
            "shear": (0.0, 0.0),     # ○ 0~2°   : 카메라 tilt 시뮬
            "perspective": (0.0, 0.0000), # ○ 0~0.001 : edge‑case 보강
            "flipud": (0.0, 0.00),   # ○ 0~0.10 : 상하반전은 드뭄
            "fliplr": (0.0, 0.00),   # ○ 0~0.50 : 좌우반전은 최대 50 %
            "mosaic": (0.00, 0.00),  # ○ 0~1.00 : 대표 혼합 증강
            "mixup": (0.00, 0.00),   # ○ 0~0.30 : 30 % ↑는 bbox 혼란
            "copy_paste": (0.00, 0.00), # ○ 0~0.30 : 객체밀도↑ 시만
            "cutmix": (0.00, 0.00),  # ○ 0~0.30 : label noise ↑ 주의
            "bgr": (0.00, 0.00),     # ○ 0 or 1  : 채널스왑 on/off
        }

    else:
        raise ValueError(f"Unsupported optimizer for tuning: {cfg.optimizer}")

    return base_kwargs, search_space

# def get_default_augmentation():
#     return {
#         "hsv_h": 0.015,             # 0.0 - 1.0
#         "hsv_s": 0.7,               # 0.0 - 1.0
#         "hsv_v": 0.4,               # 0.0 - 1.0
#         "degrees": 0.0,             # 0.0 - 180
#         "translate": 0.1,           # 0.0 - 1.0
#         "scale": 0.5,               # >=0.0
#         "shear": 0.0,               # -180 - +180
#         "perspective": 0.0,         # 0.0 - 0.001
#         "flipud": 0.0,              # 0.0 - 1.0
#         "fliplr": 0.5,              # 0.0 - 1.0
#         "bgr": 0.0,                 # 0.0 - 1.0
#         "mosaic": 1.0,              # 0.0 - 1.0
#         "mixup": 0.0,               # 0.0 - 1.0
#         "cutmix": 0.0,              # 0.0 - 1.0
#         "copy_paste": 0.0,          # 0.0 - 1.0
#         "copy_paste_mode": "flip",  # flip or mixup
#         "auto_augment": None,       # randaugment, autoaugment 또는 augmix
#         "erasing": 0.4,             # 0.0 - 0.9
#     }
