import os
import json
import numpy as np
from tqdm import tqdm
import shutil
from typing import Dict, List, Tuple


categories = [
    {
        "id": 0,
        "supercategory": "person",
        "name": "person"
    },
    {
        "id": 1,
        "supercategory": "head",
        "name": "head"
    },
    {
        "id": 2,
        "supercategory": "face",
        "name": "face"
    },
    {
        "id": 3,
        "supercategory": "lefthand",
        "name": "lefthand"
    },
    {
        "id": 4,
        "supercategory": "righthand",
        "name": "righthand"
    },
    {
        "id": 5,
        "supercategory": "leftfoot",
        "name": "leftfoot"
    },
    {
        "id": 6,
        "supercategory": "rightfoot",
        "name": "rightfoot"
    }
]

def process_coco_human_parts(anno_data_file: str, dest_anno_file: str = None) -> None:

    with open(anno_data_file, 'r') as f:
        anno_data = json.load(f)

    new_anno_data = {
        "categories": categories,
        "annotations": [],
        "images": anno_data["images"],
    }

    new_annos = []
    new_anno_count = 1
    for anno in anno_data["annotations"]:
        anno_hier = anno["hier"]
        del anno["hier"]
        anno["id"] = new_anno_count
        anno["category_id"] = 0
        new_anno_count += 1
        new_annos.append(anno)
        for part_label in range(1, 7):
            hier_index = (part_label - 1) * 5
            part_bbox = anno_hier[hier_index:hier_index+4]
            part_ignore = anno_hier[hier_index+4]
            if part_ignore != 0:
                part_area = int(np.abs(part_bbox[2] - part_bbox[0]) * np.abs(part_bbox[3] - part_bbox[1]))
                part_anno = anno.copy()
                part_anno["bbox"] = part_bbox
                part_anno["area"] = part_area
                part_anno["category_id"] = part_label
                part_anno["id"] = new_anno_count
                new_anno_count += 1
                new_annos.append(part_anno)
    
    new_anno_data["annotations"] = new_annos

    with open(dest_anno_file, 'w') as f:
        json.dump(new_anno_data, f)

if __name__ == '__main__':
    anno_file = "E:/datasets/coco_dataset/humanparts_coco_format/person_humanparts_train2017.json"
    dest_anno_file = "E:/datasets/coco_dataset/humanparts_coco_format/person_humanparts_train2017_coco_format.json"
    
    process_coco_human_parts(anno_file, dest_anno_file)