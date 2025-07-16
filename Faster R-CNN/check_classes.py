# check_classes.py

import json
from pathlib import Path
from tqdm import tqdm

# config 파일과 동일한 경로를 사용합니다.
ANNOTATION_DIR = Path("./data/train_annotations")

def find_max_category_id():
    # 모든 json 파일 목록을 가져옵니다.
    json_paths = list(ANNOTATION_DIR.glob('**/*.json'))
    
    if not json_paths:
        print(f"오류: '{ANNOTATION_DIR}' 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
        return

    max_id = -1
    all_ids = set()

    print(f"총 {len(json_paths)}개의 JSON 파일에서 category_id를 확인합니다...")
    
    # 각 파일을 순회하며 category_id를 찾습니다.
    for path in tqdm(json_paths):
        with open(path, 'r') as f:
            data = json.load(f)
            for ann in data.get('annotations', []):
                cat_id = ann.get('category_id')
                if cat_id is not None:
                    all_ids.add(cat_id)

    if not all_ids:
        print("어노테이션에서 유효한 category_id를 찾지 못했습니다.")
        return
        
    # 찾은 ID 중 최댓값을 구합니다.
    max_id = max(all_ids)
    
    print("\n--- 결과 ---")
    print(f"데이터셋에서 발견된 최대 category_id: {max_id}")
    # 모델의 num_classes는 (최대 ID + 1) 이어야 합니다. (0부터 시작하므로)
    print(f"권장되는 NUM_CLASSES 값: {max_id + 1}")

if __name__ == '__main__':
    find_max_category_id()