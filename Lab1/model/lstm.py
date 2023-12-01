import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.vocab import NERVocab
from data_utils.load_data import create_ans_space

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.embedding_dim = config['lstm']['embedding_dim']
        self.hidden_dim = config['lstm']['hidden_dim']
        self.num_layers = config['lstm']['num_layers']
        self.dropout = config['lstm']['dropout'] 
        self.vocab = NERVocab(config)
        self.POS_space,_=create_ans_space(config)
    
        self.embedding = nn.Embedding(self.vocab.vocab_size(), self.embedding_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                            num_layers=self.num_layers, batch_first=True, 
                            dropout=self.dropout) 
        self.dense = nn.Linear(self.hidden_dim,self.POS_space+1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_layer(x)
        lstm_out, _ = self.lstm(x)
        out = self.dense(lstm_out)
        return F.log_softmax(out,dim=-1)
    
class LSTM_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = LSTM(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        if labels is not None:
            logits = self.lstm(inputs)
            loss = self.loss_fn(logits.view(-1,locals.size(-1)), labels.view(-1))
            return logits, loss
        else:
            logits = self.lstm(inputs)
            return logits
