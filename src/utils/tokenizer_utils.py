import torch

def get_tokenizer_from_vocab(vocab_path):
    """Create a simple tokenizer from a vocabulary file"""
    if not torch.io.is_file(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    vocab = torch.load(vocab_path)
    
    class CharTokenizer:
        def __init__(self, char_to_idx, idx_to_char, vocab_size):
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
            self.vocab_size = vocab_size
        
        def encode(self, text):
            return [self.char_to_idx.get(char, self.vocab_size-1) for char in text]
        
        def decode(self, ids):
            return ''.join([self.idx_to_char.get(id, '[UNK]') for id in ids])
    
    if 'char_to_idx' in vocab and 'idx_to_char' in vocab:
        return CharTokenizer(
            vocab['char_to_idx'],
            vocab['idx_to_char'],
            vocab['vocab_size']
        )
    
    raise ValueError("Unexpected vocabulary structure")
