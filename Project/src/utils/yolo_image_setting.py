import json

import shutil
from pathlib import Path

def copy_images(image_source, label_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    
    for label_file in label_dir.glob("*.txt"):
        image_name = label_file.stem + ".png"
        source_img = image_source / image_name
        target_img = output_dir / image_name
        
        if source_img.exists():
            shutil.copy2(source_img, target_img)
            copied += 1
    
    return copied

def setup_yolo_images(data_path):
    base = Path(data_path)
    image_source = base / "train_images"
    
    train_copied = copy_images(image_source, base / "labels" / "train", base / "images" / "train")
    val_copied = copy_images(image_source, base / "labels" / "val", base / "images" / "val")
    
    print(f"Train: {train_copied}, Val: {val_copied}")

# 1. COCO-style category_id → YOLO 인덱스 매핑 생성
def create_class_mappings(category_id_map):
    """
    COCO-style category_id → class_name 매핑을 기반으로,
    YOLO 학습에 필요한 클래스 인덱스 매핑을 생성합니다.

    Args:
        category_id_map (dict): {original_category_id: class_name}
            - COCO 등에서 사용되는 원래의 category_id와 클래스 이름 매핑

    Returns:
        id_to_index (dict): {original_category_id → YOLO class index}
            - 원래의 category_id를 YOLO에서 사용하는 0부터 시작하는 인덱스로 변환

        index_to_id (dict): {YOLO class index → original_category_id}
            - YOLO 인덱스를 원래의 category_id로 되돌리는 매핑

        index_to_name (dict): {YOLO class index → class_name}
            - YOLO 인덱스를 사람이 읽을 수 있는 클래스 이름으로 매핑

        names_list (list): [class_name, ...]
            - YOLO의 data.yaml 파일에 들어갈 클래스 이름 리스트
            - YOLO 인덱스 순서대로 정렬됨
    """
    original_ids = sorted(category_id_map.keys())  # category_id 정렬
    id_to_index = {orig_id: idx for idx, orig_id in enumerate(original_ids)}  # COCO → YOLO index
    index_to_id = {idx: orig_id for orig_id, idx in id_to_index.items()}      # YOLO index → COCO
    index_to_name = {idx: category_id_map[orig_id] for idx, orig_id in index_to_id.items()}  # YOLO index → name
    names_list = [category_id_map[orig_id] for orig_id in original_ids]       # YOLO index 순서대로 이름 리스트

    return id_to_index, index_to_id, index_to_name, names_list


def weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir):
    mc = 0
    for label_file in label_dir.glob("*.txt"):
        kkkk = [ii for ii in t_ls if ii in str(label_file)]
        if len(kkkk) > 0:
            kkkk2 = [ii for ii in ig_ls if ii in str(label_file)]
            if len(kkkk2) > 0:
                continue
            image_name = label_file.stem + ".png"
            t_image_name = label_file.stem + f"_{idx}.png"

            source_img = image_source / image_name
            target_img = output_dir / t_image_name

            t_label_name = label_file.stem + f"_{idx}.txt"
            target_label = label_dir / t_label_name

            if source_img.exists():
                shutil.copy2(source_img, target_img)

                shutil.copy2(label_file, target_label)

                mc += 1
    print("make count", idx, mc)


def augumentation():
    annotation_dir = Path() / data_path / "train_annotations"

    json_path = Path() / "Project" / "data_csv" / "mappings.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    mappings['image_id_map'] = {
        int(k): v for k, v in mappings.get('image_id_map', {}).items()
    }

    mappings['category_id_map'] = {
        int(k): v for k, v in mappings.get('category_id_map', {}).items()
    }

    id_to_index, index_to_id, index_to_name, names_list = create_class_mappings(mappings['category_id_map'])

    ann_label_temp = {}
    for ann_file in annotation_dir.glob("**/*.json"):
        with open(ann_file, "r", encoding="utf-8-sig") as f:
            ann = json.load(f)
            key = (ann["images"][0]["drug_N"]).split("-")[1]

            if key not in ann_label_temp:
                for i, name in enumerate(names_list):
                    if name == ann["categories"][0]["name"]:
                        ann_label_temp[key] = i
                        break

    base = Path(data_path)
    image_source = base / "train_images"
    label_dir = base / "labels" / "train"
    output_dir = base / "images" / "train"
    v_label_dir = base / "labels" / "val"
    v_output_dir = base / "images" / "val"

    idx = 1
    t_ls = [key for key, v in ann_label_temp.items() if v not in [3]]
    ig_ls = [key for key, v in ann_label_temp.items() if v in [3]]
    weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir)
    weight_copy(idx, t_ls, ig_ls, image_source, v_label_dir, v_output_dir)

    idx += 1
    ig_ls = ig_ls + [key for key, v in ann_label_temp.items() if v in [2]]
    weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir)
    weight_copy(idx, t_ls, ig_ls, image_source, v_label_dir, v_output_dir)

    idx += 1
    ig_ls = ig_ls + [key for key, v in ann_label_temp.items() if v in [0, 1]]
    weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir)
    weight_copy(idx, t_ls, ig_ls, image_source, v_label_dir, v_output_dir)

    idx += 1
    weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir)
    weight_copy(idx, t_ls, ig_ls, image_source, v_label_dir, v_output_dir)

    idx += 1
    weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir)
    weight_copy(idx, t_ls, ig_ls, image_source, v_label_dir, v_output_dir)

    idx += 1
    ig_ls = ig_ls + [key for key, v in ann_label_temp.items() if v in [4]]
    weight_copy(idx, t_ls, ig_ls, image_source, label_dir, output_dir)
    weight_copy(idx, t_ls, ig_ls, image_source, v_label_dir, v_output_dir)


if __name__ == "__main__":
    data_path = "./data/ai03-level1-project"
    setup_yolo_images(data_path)
    
    augumentation()
