import os
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import torch

class data_Collator:
    def __init__(self, config: Dict):
        self.config = config
    def __call__(self, raw_batch_dict):
        return {
            #'text':[scanerr(ann["sentence"]) for ann in raw_batch_dict],
            'text':[ann["sentence"] for ann in raw_batch_dict],
            'labels': torch.tensor([ann["label"] for ann in raw_batch_dict],
                dtype=torch.int64
            ),

        }

def createDataCollator(config: Dict) -> data_Collator:
    collator = data_Collator(config)
    return collator



