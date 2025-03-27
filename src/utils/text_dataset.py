# src/utils/text_dataset.py
import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
import random

logger = logging.getLogger('edgeformer')

class TextDataset(Dataset):
    """Dataset for language modeling tasks with EdgeFormer."""
    
    def __init__(self, text_data, seq_length=128, tokenizer=None):
        """
        Args:
            text_data: Raw text corpus, path to text file, or pre-tokenized data (list/tensor)
            seq_length (int): Sequence length for training
            tokenizer: Optional tokenizer (if None, uses a simple character-based tokenization)
        """
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.vocab_size = 0
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Handle pre-tokenized data
        if isinstance(text_data, (list, torch.Tensor)):
            logger.info("Using provided pre-tokenized data")
            self.tokenized = text_data if isinstance(text_data, torch.Tensor) else torch.tensor(text_data)
            self.data_size = len(self.tokenized) - seq_length
            logger.info(f"Pre-tokenized dataset contains {self.data_size} sequences of length {seq_length}")
            return
            
        # Load the text data
        if os.path.isfile(text_data):
            logger.info(f"Loading text from file: {text_data}")
            with open(text_data, 'r', encoding='utf-8', errors='ignore') as f:
                self.raw_text = f.read()
        else:
            logger.info("Using provided text data directly")
            self.raw_text = text_data
            
        logger.info(f"Text corpus size: {len(self.raw_text)} characters")
        
        # Tokenize the text
        if tokenizer is None:
            # Simple character-level tokenization
            logger.info("Using character-level tokenization")
            self.vocab = sorted(list(set(self.raw_text)))
            self.vocab_size = len(self.vocab)
            self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
            
            logger.info(f"Vocabulary size: {self.vocab_size}")
            
            # Convert text to token indices
            self.tokenized = [self.char_to_idx[c] for c in self.raw_text]
        else:
            # Use the provided tokenizer
            logger.info("Using provided tokenizer")
            encoded = tokenizer(self.raw_text)
            self.tokenized = encoded["input_ids"]
            self.vocab_size = tokenizer.vocab_size
            
        self.data_size = len(self.tokenized) - seq_length
        logger.info(f"Dataset contains {self.data_size} sequences of length {seq_length}")
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Get sequence at position idx
        input_sequence = self.tokenized[idx:idx + self.seq_length]
        target_sequence = self.tokenized[idx + 1:idx + self.seq_length + 1]
        
        # Convert to tensors if needed
        if not isinstance(input_sequence, torch.Tensor):
            inputs = torch.tensor(input_sequence, dtype=torch.long)
            targets = torch.tensor(target_sequence, dtype=torch.long)
        else:
            inputs = input_sequence
            targets = target_sequence
        
        return {
            "input_ids": inputs,
            "labels": targets
        }

def create_wikitext_dataset(seq_length=128, tokenizer=None, split="train"):
    """
    Download and create a dataset from WikiText.
    
    Args:
        seq_length: Sequence length for training
        tokenizer: Optional tokenizer
        split: "train", "validation" or "test"
        
    Returns:
        TextDataset instance
    """
    try:
        from datasets import load_dataset
        
        logger.info(f"Loading WikiText-2 dataset ({split} split)")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")[split]
        
        # Concatenate all texts
        text = " ".join(dataset["text"])
        
        return TextDataset(text, seq_length, tokenizer)
    
    except ImportError:
        logger.error("Could not import 'datasets' library. Install with: pip install datasets")
        raise

def get_data_loaders(dataset, batch_size=4, val_split=0.1, shuffle=True):
    """
    Create training and validation data loaders from a dataset.
    
    Args:
        dataset: TextDataset instance
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        
    Returns:
        train_loader, val_loader
    """
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Split dataset into {train_size} training and {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Created data loaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    return train_loader, val_loader