"""Embeddings module for EdgeFormer."""
import torch
import torch.nn as nn

class EdgeFormerEmbeddings(nn.Module):
    """Embeddings for the EdgeFormer model."""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids, position_ids=None):
        """Forward pass."""
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.dropout(embeddings)
        
        return embeddings