
# utils/data_utils.py

import pandas as pd
import json

def load_filtered_df(cfg):
    return pd.read_csv(cfg.filtered_annotation_csv_path)

def load_mappings(cfg):
    with open(cfg.mappings_json_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    mappings['image_id_map'] = {
        int(k): v for k, v in mappings.get('image_id_map', {}).items()
    }

    mappings['category_id_map'] = {
        int(k): v for k, v in mappings.get('category_id_map', {}).items()
    }

    return mappings