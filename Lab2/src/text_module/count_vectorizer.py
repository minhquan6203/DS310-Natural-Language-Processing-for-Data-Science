import torch
from torch import nn
from data_utils.vocab import Vocab
from mask.masking import generate_padding_mask

class CountVectorizer(nn.Module):
    def __init__(self, config):
        super(CountVectorizer, self).__init__()
        self.vocab = Vocab(config)
        self.proj = nn.Linear(self.vocab.vocab_size(), config["text_embedding"]["d_model"])

    def forward(self, input_texts):
        count_vectors = []
        for input_text in input_texts:
            word_counts = torch.zeros(self.vocab.vocab_size())
            for word in input_text.split():
                word_counts[self.vocab.word_to_idx.get(word,self.vocab.word_to_idx['[UNK]'])] += 1
            count_vectors.append(word_counts)
        
        count_vectors = torch.stack(count_vectors, dim=0)  # Xếp các word_counts thành một tensor
        count_vectors = count_vectors.to(self.proj.weight.device)  # Chuyển đổi sang cùng device với self.proj
        count_vectors = self.proj(count_vectors).unsqueeze(1)
        padding_mask = generate_padding_mask(count_vectors, padding_idx=0)
        return count_vectors, padding_mask

