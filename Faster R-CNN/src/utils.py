# src/utils.py

import json
from pathlib import Path
from tqdm import tqdm
from . import config

def get_class_mapping(recreate=False):
    """
    category_id와 약제 이름(dl_name)을 연결하는 매핑 파일을 로드하거나 생성합니다.
    매핑 파일은 outputs/class_mapping.json에 저장됩니다.

    Args:
        recreate (bool): True일 경우, 기존 매핑 파일을 무시하고 새로 생성합니다.

    Returns:
        dict: { "category_id": "약제이름", ... } 형식의 딕셔너리
    """
    mapping_path = config.OUTPUT_DIR / "class_mapping.json"

    # 파일이 존재하고, 새로 만들 필요가 없으면 기존 파일을 로드
    if not recreate and mapping_path.exists():
        print("기존 클래스 매핑 파일을 로드합니다...")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            # JSON은 키를 문자열로 저장하므로, 불러온 그대로 사용
            return json.load(f)

    # 매핑 파일 생성 시작
    print("새로운 클래스 매핑 파일을 생성합니다...")
    annotation_dir = Path(config.TRAIN_ANNOTATION_DIR)
    json_paths = list(annotation_dir.glob('**/*.json'))

    id_to_name = {}
    for path in tqdm(json_paths, desc="JSON 파일 스캔 중"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # JSON 파일 내에서 category_id와 dl_name 추출
            # 각 JSON 파일의 정보가 하나의 약품 종류에 해당한다고 가정
            category_id = data.get('annotations', [{}])[0].get('category_id')
            drug_name = data.get('images', [{}])[0].get('dl_name')

            if category_id is not None and drug_name:
                # 키는 문자열로 저장해야 JSON과 호환됨
                id_to_name[str(category_id)] = drug_name

    # 생성된 매핑 파일을 저장
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_name, f, ensure_ascii=False, indent=4)
        
    print(f"매핑 파일 생성 완료: {mapping_path}")
    return id_to_name