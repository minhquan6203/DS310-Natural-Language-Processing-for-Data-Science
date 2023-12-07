from model.bert import Bert_Model
from model.lstm import LSTM_Model

def build_model(config):
    if config['model']['type_model']=='lstm':
        return LSTM_Model(config)
    if config['model']['type_model']=='bert':
        return Bert_Model(config)