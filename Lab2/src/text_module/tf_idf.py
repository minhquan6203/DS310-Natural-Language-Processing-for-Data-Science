import torch
import torch.nn as nn
from mask.masking import generate_padding_mask
from data_utils.vocab import Vocab
class IDFVectorizer(nn.Module):
    def __init__(self, config):
        super(IDFVectorizer, self).__init__()
        self.vocab = Vocab(config)
        self.all_word, self.word_count = self.vocab.all_word()
        self.idf_vector = self.compute_idf_vector()
        self.proj = nn.Linear(self.vocab.vocab_size(), config["text_embedding"]["d_model"])
    
    def compute_idf_vector(self):
        idf_vector = torch.zeros(self.vocab.vocab_size())
        for i, word in enumerate(self.all_word):
            if word in self.word_count:
                idf_value = torch.log(torch.tensor(len(self.word_count) / self.word_count[word]))
                idf_vector[i] = idf_value
        return idf_vector
    
    def compute_tf_vector(self, input_text):
        tf_vector = torch.zeros(self.vocab.vocab_size())
        total_words = len(input_text.split())
        
        for word in input_text.split():
            tf_vector[self.vocab.word_to_idx.get(word,self.vocab.word_to_idx['[UNK]'])] += 1
        return tf_vector / total_words
    
    def forward(self, input_texts):
        tf_idf_vectors = []
        for input_text in input_texts:
            tf_vector = self.compute_tf_vector(input_text)
            tf_idf_vectors.append(tf_vector*self.idf_vector)
        tf_idf_vectors = torch.stack(tf_idf_vectors, dim=0)
        embedding = self.proj(tf_idf_vectors.to(self.proj.weight.device)).unsqueeze(1)
        padding_mask = generate_padding_mask(embedding, padding_idx=0)
        return embedding, padding_mask
