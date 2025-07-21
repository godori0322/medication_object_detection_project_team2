import numpy as np

def coco_to_voc(boxes):
    if len(boxes) == 0:
        return boxes
    return np.array([[x, y, x + w, y + h] for x, y, w, h in boxes])