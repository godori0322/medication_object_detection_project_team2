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

if __name__ == "__main__":
    data_path = "./data/ai03-level1-project"
    setup_yolo_images(data_path)