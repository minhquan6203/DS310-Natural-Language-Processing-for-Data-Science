import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from data_utils.load_data_bert import create_ans_space

#design for phobert, xlm-roberta, videberta, bartpho, pretrained in english also supported 
class Text_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Embedding,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.embedding = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"])
        # freeze all parameters of pretrained model
        if config["text_embedding"]["freeze"]:
            for param in self.embedding.parameters():
                param.requires_grad = False
        
        if config['text_embedding']['freeze']==False and config['text_embedding']['use_lora']==True:
            lora_config = LoraConfig(
                r=config['text_embedding']['lora_r'],
                lora_alpha=config['text_embedding']['lora_alpha'],
                # target_modules=config['text_embedding']['lora_target_modules'],
                lora_dropout=config['text_embedding']['lora_dropout'],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            # self.embedding = prepare_model_for_int8_training(self.embedding)
            self.embedding = get_peft_model(self.embedding, lora_config)
        self.POS_space,self.Tag_space=create_ans_space(config)
        self.proj = nn.Linear(config["text_embedding"]['d_features'], len(self.Tag_space)+1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.padding = config["tokenizer"]["padding"]
        self.truncation = config["tokenizer"]["truncation"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"]
        self.max_length = config["tokenizer"]["max_input_length"]

    def forward(self, text: List[str]):
        inputs = self.tokenizer(
            text,
            padding = self.padding,
            max_length = self.max_length,
            truncation = self.truncation,
            return_tensors = 'pt',
            return_attention_mask = self.return_attention_mask,
        ).to(self.device)
        features = self.embedding(**inputs).last_hidden_state
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out

class Bert_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = Text_Embedding(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, inputs, labels=None):
        if labels is not None:
            logits = self.bert(inputs)
            labels=labels.to(self.device)
            loss = self.loss_fn(logits.view(-1,logits.size(-1)), labels.view(-1))
            return logits, loss
        else:
            logits = self.bert(inputs)
            return logits

