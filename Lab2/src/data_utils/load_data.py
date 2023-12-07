import os
from typing import Dict
from datasets import load_dataset
import numpy as np

def loadDataset(config: Dict) -> Dict:
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "val": os.path.join(config["data"]["dataset_folder"], config["data"]["val_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["test_dataset"])
        }
    )

    #answer_space = [0, 1, 2]
    #answer_space =['Sadness' 'Surprise' 'Disgust' 'Fear' 'Anger' 'Other' 'Enjoyment']
    answer_space = list(np.unique(dataset['train']['sentiment']))
    dataset = dataset.map(
        lambda examples: {'label': [answer_space.index(ans) for ans in examples['sentiment']]},
        batched=True
    )
    
    dataset = dataset.shuffle(123)  # Xáo trộn tập dữ liệu
    
    return {
        "dataset": dataset,
        "answer_space": answer_space
    }
