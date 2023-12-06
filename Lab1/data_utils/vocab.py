import os
from typing import List, Dict, Optional
from datasets import load_dataset
import os
import numpy as np
import pandas as pd

class NERVocab:
    def __init__(self, config: Dict):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.data_folder=config['data']['dataset_folder']
        self.train_set=config["data"]["train_dataset"]
        self.val_set=config["data"]["val_dataset"]
        self.test_set=config["data"]["test_dataset"]

        self.build_vocab()

    def all_word(self):
        dataset = load_dataset(
            "csv", 
            data_files={
                "train": os.path.join(self.data_folder, self.train_set),
                "val": os.path.join(self.data_folder, self.val_set),
                "test": os.path.join(self.data_folder, self.test_set)
            }
        )

        word_counts = {}

        for data_file in dataset.values():
            try:
                for word in data_file['Word']:
                        # word=word.lower()
                        if word not in word_counts:
                            word_counts[word] = 1
                        else:
                            word_counts[word] += 1
            except:
                pass

        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
        vocab = list(sorted_word_counts.keys())
        if '<UNK>' not in vocab:
            vocab.append("<UNK>")
        if '[SEP]' not in vocab:
            vocab.append("[SEP]")
        if '[CLS]' not in vocab:
            vocab.append("[CLS]")

        return vocab, sorted_word_counts

        
    def build_vocab(self):
        all_word,_=self.all_word()
        self.word_to_idx = {word: idx+1  for idx, word in enumerate(all_word)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def convert_tokens_to_ids(self, tokens):
        return [self.word_to_idx.get(token, 0) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_word[idx] for idx in ids]

    def vocab_size(self):
        return len(self.word_to_idx)+1

    def pad_token_id(self):
        return 0  # ID for the padding token



def create_ans_space(config: Dict):
    train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
    df_train=pd.read_csv(train_path)
    POS_space = list(np.unique(df_train['POS']))
    Tag_space = list(np.unique(df_train['Tag']))
    POS_to_index = {label: index for index, label in enumerate(POS_space)}
    df_train['POS'] =df_train['POS'].map(POS_to_index)

    Tag_to_index = {label: index for index, label in enumerate(Tag_space)}
    df_train['Tag'] =df_train['Tag'].map(Tag_to_index)
    return POS_space, Tag_space