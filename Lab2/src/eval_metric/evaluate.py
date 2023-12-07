from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import wordnet

class ScoreCalculator:
    def __init__(self, answer_space: List[str]):
        self.answer_space = answer_space

    def compute_metrics(self, eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_tuple
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='macro')
        }