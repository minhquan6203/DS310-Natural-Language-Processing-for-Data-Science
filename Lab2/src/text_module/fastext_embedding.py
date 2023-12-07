import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.vocab import Vocab
import fasttext
from torch.nn.utils.rnn import pad_sequence
from mask.masking import generate_padding_mask
from typing import List, Dict, Optional
import numpy as np

class Fastext_Embedding(nn.Module):
    def __init__(self, config):
        super(Fastext_Embedding, self).__init__()
        self.embedding_dim = config['text_embedding']['d_features']
        self.vocab = Vocab(config)
        self.embedding = fasttext.load_model('./cc.vi.300.bin')
        self.dropout = nn.Dropout(config['text_embedding']['dropout'])
        self.gelu = nn.GELU()
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.proj = nn.Linear(self.embedding_dim, config["text_embedding"]["d_model"])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value):
        if max_len == 0:
            tensor = torch.zeros((0, tensor.shape[-1]))
        else:
            pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value)
            tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        return tensor

    def forward(self, input_texts):
        features=[]
        for text in input_texts:
            text_feature = [torch.tensor(self.embedding.get_sentence_vector(word.lower())) for word in text.split()[:self.max_length]]
            text_feature=torch.stack(text_feature)
            text_feature = self.pad_tensor(text_feature,self.max_length,0)
            features.append(text_feature)
        
        features=torch.stack(features).to(self.device)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask