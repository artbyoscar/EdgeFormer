import torch
import numpy as np

class SimpleTokenizer:
    """
    A basic character-level tokenizer for EdgeFormer.
    """
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        self.is_fitted = False
        self.pad_token_id = 0
        self.unk_token_id = 1
        
    def fit(self, texts):
        """
        Fit the tokenizer on a list of texts to create the vocabulary.
        
        Args:
            texts: List of strings to extract vocabulary from
        """
        # Start with special tokens
        self.char_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Extract unique characters from texts
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Add characters to vocabulary
        for char in sorted(all_chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
        
        # Create reverse mapping
        self.id_to_char = {id: char for char, id in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.is_fitted = True
        
        return self
    
    def encode(self, text):
        """
        Encode a string into token IDs.
        
        Args:
            text: String to encode
            
        Returns:
            List of token IDs
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        return [self.char_to_id.get(char, self.unk_token_id) for char in text]
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs back to a string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding")
        
        return ''.join(self.id_to_char.get(token_id, '<UNK>') for token_id in token_ids 
                       if token_id not in [self.pad_token_id])
    
    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size
    
    def save(self, path):
        """Save tokenizer to a file"""
        torch.save(self, path)
    
    @staticmethod
    def load(path):
        """Load tokenizer from a file"""
        return torch.load(path)
