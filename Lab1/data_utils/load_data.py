import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils.vocab import NERVocab
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from typing import List, Dict, Optional
class NERDataset(Dataset):
    def __init__(self, df, vocab, max_len, with_labels=True):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.with_labels = with_labels
        self.data= self.process_data()

    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data['inputs'][idx], dtype = torch.int32)
        if self.with_labels:
            labels = torch.tensor(self.data['targets'][idx], dtype=torch.long)
            return {'inputs': inputs, 'labels': labels}
        else:
            return {'inputs': inputs}

    def process_data(self):
        sentences = self.get_sentences()
        X = [
            [
                self.vocab.word_to_idx.get(w[0], self.vocab.word_to_idx['<UNK>']) 
                for w in s[:self.max_len]
            ] 
            for s in sentences
        ]
        X = pad_sequence(
            [torch.tensor(x, dtype=torch.int32) for x in X], 
            padding_value=float(self.vocab.pad_token_id()), 
            batch_first=True
        )
        if self.with_labels:
            y = [
                [w[2] for w in s[:self.max_len]] 
                for s in sentences
            ]

            y = pad_sequence([torch.tensor(label, dtype=torch.long) for label in y], 
                            padding_value=0, 
                            batch_first=True)

            return {'inputs': X, 'targets': y}
        else:
            return {'inputs': X}



    def get_sentences(self):
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        grouped = self.df.groupby("Sentence #").apply(agg_func)
        return [s for s in grouped]

class Get_Loader:
    def __init__(self, config):
        self.vocab = NERVocab(config)

        self.train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
        self.train_batch=config['train']['per_device_train_batch_size']

        self.val_path=os.path.join(config['data']['dataset_folder'],config['data']['val_dataset'])
        self.val_batch=config['train']['per_device_valid_batch_size']

        self.test_path=os.path.join(config['inference']['test_dataset'])
        self.test_batch=config['inference']['batch_size']
        
        self.max_len = config['tokenizer']['max_input_length']
        self.with_labels= config['inference']['with_labels']

    def load_data_train_dev(self):
        df_train = pd.read_csv(self.train_path, encoding="latin1")
        df_val = pd.read_csv(self.val_path, encoding="latin1")

        POS_space = list(np.unique(df_train['POS']))
        Tag_space = list(np.unique(df_train['Tag']))
        POS_to_index = {label: index+1 for index, label in enumerate(POS_space)}
        Tag_to_index = {label: index+1 for index, label in enumerate(Tag_space)}
        df_train['POS'] =df_train['POS'].map(POS_to_index)        
        df_train['Tag'] =df_train['Tag'].map(Tag_to_index)

        df_val['POS'] =df_val['POS'].map(POS_to_index)
        df_val['Tag'] =df_val['Tag'].map(Tag_to_index)

        train_data = NERDataset(df_train, self.vocab, self.max_len)
        val_data = NERDataset(df_val, self.vocab, self.max_len)
        train_loader = DataLoader(train_data, batch_size=self.train_batch, shuffle=True)
        dev_loader = DataLoader(val_data, batch_size=self.val_batch, shuffle=True)

        return train_loader, dev_loader
    
    def load_data_test(self):
        df_test = pd.read_csv(self.test_path, encoding="latin1")
        POS_space = list(np.unique(df_test['POS']))
        Tag_space = list(np.unique(df_test['Tag']))
        POS_to_index = {label: index+1 for index, label in enumerate(POS_space)}
        Tag_to_index = {label: index+1 for index, label in enumerate(Tag_space)}
        df_test['POS'] =df_test['POS'].map(POS_to_index)        
        df_test['Tag'] =df_test['Tag'].map(Tag_to_index)
        test_data = NERDataset(df_test, self.vocab, self.max_len, self.with_labels)
        test_loader = DataLoader(test_data, batch_size=self.test_batch, shuffle=False)
        return test_loader

def create_ans_space(config: Dict):
    train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
    df_train=pd.read_csv(train_path)
    POS_space = list(np.unique(df_train['POS']))
    Tag_space = list(np.unique(df_train['Tag']))
    POS_to_index = {label: index+1 for index, label in enumerate(POS_space)}
    df_train['POS'] =df_train['POS'].map(POS_to_index)

    Tag_to_index = {label: index+1 for index, label in enumerate(Tag_space)}
    df_train['Tag'] =df_train['Tag'].map(Tag_to_index)
    return POS_space, Tag_space