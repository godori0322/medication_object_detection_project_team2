import json
from pathlib import Path
from tqdm import tqdm
from .. import config

def create_category_mapping(recreate=False):
    """
    category_id를 0부터 시작하는 연속 인덱스로 매핑하는 딕셔너리를 생성합니다.
    
    Args:
        recreate (bool): True일 경우, 기존 매핑 파일을 무시하고 새로 생성합니다.
    
    Returns:
        tuple: (id_to_index, index_to_id, num_classes)
            - id_to_index: {category_id: index} 매핑
            - index_to_id: {index: category_id} 역매핑  
            - num_classes: 총 클래스 수
    """
    mapping_path = config.OUTPUT_DIR / "id_mapping.json"
    
    # 파일이 존재하고, 새로 만들 필요가 없으면 기존 파일을 로드
    if not recreate and mapping_path.exists():
        print("기존 ID 매핑 파일을 로드합니다...")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            id_to_index = {int(k): v for k, v in data['id_to_index'].items()}
            index_to_id = {int(k): v for k, v in data['index_to_id'].items()}
            return id_to_index, index_to_id, data['num_classes']
    
    # 매핑 파일 생성 시작
    print("새로운 ID 매핑 파일을 생성합니다...")
    annotation_dir = Path(config.TRAIN_ANNOTATION_DIR)
    json_paths = list(annotation_dir.glob('**/*.json'))
    
    # 모든 고유한 category_id 수집
    unique_ids = set()
    for path in tqdm(json_paths, desc="고유 category_id 수집 중"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for ann in data.get('annotations', []):
                cat_id = ann.get('category_id')
                if cat_id is not None:
                    unique_ids.add(cat_id)
    
    # 정렬된 고유 ID들을 0부터 시작하는 연속 인덱스로 매핑
    sorted_ids = sorted(unique_ids)
    id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted_ids)}
    index_to_id = {idx: cat_id for idx, cat_id in enumerate(sorted_ids)}
    num_classes = len(sorted_ids)
    
    # 매핑 정보 저장
    mapping_data = {
        'id_to_index': {str(k): v for k, v in id_to_index.items()},
        'index_to_id': {str(k): v for k, v in index_to_id.items()},
        'num_classes': num_classes,
        'original_ids': sorted_ids
    }
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=4)
    
    print(f"ID 매핑 생성 완료: {num_classes}개 클래스")
    print(f"매핑 파일 저장: {mapping_path}")
    
    return id_to_index, index_to_id, num_classes

def map_category_id_to_index(category_id, id_to_index_mapping):
    """
    원본 category_id를 연속 인덱스로 변환합니다.
    
    Args:
        category_id (int): 원본 category_id
        id_to_index_mapping (dict): category_id -> index 매핑
        
    Returns:
        int: 연속 인덱스 (0부터 시작)
    """
    return id_to_index_mapping.get(category_id, -1)

def map_index_to_category_id(index, index_to_id_mapping):
    """
    연속 인덱스를 원본 category_id로 변환합니다.
    
    Args:
        index (int): 연속 인덱스
        index_to_id_mapping (dict): index -> category_id 매핑
        
    Returns:
        int: 원본 category_id
    """
    return index_to_id_mapping.get(index, -1)