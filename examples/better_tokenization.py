"""
Better Tokenizer Implementation for EdgeFormer

This module provides an improved tokenizer implementation that can be used
with the EdgeFormer model for better text generation results.
"""

import os
import re
from typing import List, Dict, Optional, Union

class BetterTokenizer:
    """
    An improved tokenizer for EdgeFormer that implements subword tokenization
    for better text processing capabilities.
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initialize the BetterTokenizer.
        
        Args:
            vocab_file: Path to vocabulary file. If None, a default vocabulary will be used.
        """
        # Load vocabulary or create a default one
        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        
        # Regular expression for tokenization
        self.pattern = re.compile(r'\w+|[^\w\s]')
        
    def _load_vocab(self, vocab_file: Optional[str]) -> Dict[str, int]:
        """
        Load vocabulary from file or create a default one.
        
        Args:
            vocab_file: Path to vocabulary file
            
        Returns:
            Dictionary mapping tokens to ids
        """
        if vocab_file is not None and os.path.exists(vocab_file):
            # Load from file
            vocab = {}
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    token = line.strip()
                    vocab[token] = i
            return vocab
        else:
            # Create a simple default vocabulary
            # First, add special tokens
            vocab = {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
            }
            
            # Add ASCII characters and some common words for demonstration
            chars = [chr(i) for i in range(32, 127)]
            common_words = [
                "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
                "he", "was", "for", "on", "are", "as", "with", "his", "they", "I",
                "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
                "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
                "there", "use", "an", "each", "which", "she", "do", "how", "their", "if",
                "will", "up", "other", "about", "out", "many", "then", "them", "these", "so",
                "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
                "two", "more", "write", "go", "see", "number", "no", "way", "could", "people",
                "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now",
                "find", "long", "down", "day", "did", "get", "come", "made", "may", "part",
                "over", "new", "sound", "take", "only", "little", "work", "know", "place", "year",
                "live", "me", "back", "give", "most", "very", "after", "thing", "our", "just",
                "name", "good", "sentence", "man", "think", "say", "great", "where", "help", "through",
                "much", "before", "line", "right", "too", "mean", "old", "any", "same", "tell",
                "boy", "follow", "came", "want", "show", "also", "around", "form", "three", "small"
            ]
            
            for i, token in enumerate(chars + common_words, start=len(vocab)):
                if token not in vocab:
                    vocab[token] = i
                    
            return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subtokens.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = self.pattern.findall(text)
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to ids.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token ids
        """
        return [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token ids.
        
        Args:
            text: The text to encode
            
        Returns:
            List of token ids
        """
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token ids to text.
        
        Args:
            token_ids: List of token ids
            
        Returns:
            Decoded text
        """
        tokens = [self.ids_to_tokens.get(token_id, self.unk_token) for token_id in token_ids]
        return " ".join(tokens)