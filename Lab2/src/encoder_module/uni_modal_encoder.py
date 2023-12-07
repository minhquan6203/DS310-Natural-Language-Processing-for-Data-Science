
import torch
from torch import nn
from typing import List, Dict, Optional
from utils.position_wise_feed_forward import PositionWiseFeedForward
from attention_module.attentions import MultiHeadAtt
from utils.positional_embbeding import SinusoidPositionalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask)
        ff = self.pwff(att)

        return ff

class UniModalEncoder(nn.Module):

    def __init__(self, config):
        super(UniModalEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config["encoder"]['d_model'])
        self.layer_norm = nn.LayerNorm(config["encoder"]['d_model'])
        self.d_model = config["encoder"]['d_model']

        self.attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])
       
    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor):
        features = self.layer_norm(features) + self.pos_embedding(features)

        for layers in self.attn_layers:
            features = layers(
                queries=features,
                keys=features,
                values=features,
                attention_mask=padding_mask
            )

        return features
