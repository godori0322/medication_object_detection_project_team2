# utils/data_utils.py

import pandas as pd
import json
from pathlib import Path


def load_filtered_df(cfg):
    csv_path = Path(cfg.base_dir) / "Project" / "data_csv" / "filtered_df.csv"
    return pd.read_csv(csv_path)

def load_mappings(cfg):
    json_path = Path(cfg.base_dir) / "Project" / "data_csv" / "mappings.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    mappings['image_id_map'] = {
        int(k): v for k, v in mappings.get('image_id_map', {}).items()
    }

    mappings['category_id_map'] = {
        int(k): v for k, v in mappings.get('category_id_map', {}).items()
    }

    return mappings