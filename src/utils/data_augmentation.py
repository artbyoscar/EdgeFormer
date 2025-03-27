# src/utils/data_augmentation.py
import random
import torch
import numpy as np
from torch.utils.data import Dataset

class TextAugmentation:
    """Text augmentation techniques for NLP training."""
    
    @staticmethod
    def random_deletion(tokens, p=0.1):
        """Randomly delete tokens with probability p."""
        if len(tokens) <= 1:
            return tokens
            
        # Don't delete too many tokens
        keep_prob = 1 - p
        keep_indices = [i for i in range(len(tokens)) if random.random() < keep_prob]
        
        # Make sure at least 1 token is kept
        if not keep_indices:
            keep_indices = [random.randint(0, len(tokens) - 1)]
            
        return [tokens[i] for i in sorted(keep_indices)]
    
    @staticmethod
    def random_swap(tokens, n=1):
        """Randomly swap n pairs of tokens."""
        if len(tokens) < 2:
            return tokens
            
        new_tokens = tokens.copy()
        for _ in range(min(n, len(tokens) // 2)):
            # Get random indices
            idx1, idx2 = random.sample(range(len(new_tokens)), 2)
            # Swap tokens
            new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
            
        return new_tokens
    
    @staticmethod
    def random_replacement(tokens, vocab, p=0.1):
        """Replace tokens with random tokens from vocabulary with probability p."""
        new_tokens = []
        for token in tokens:
            if random.random() < p:
                new_tokens.append(random.choice(vocab))
            else:
                new_tokens.append(token)
                
        return new_tokens
    
    @staticmethod
    def apply_augmentations(tokens, vocab=None, p_del=0.1, p_swap=0.1, p_repl=0.1):
        """Apply multiple augmentation techniques."""
        # Choose one augmentation randomly
        aug_type = random.choice(["none", "deletion", "swap", "replacement"])
        
        if aug_type == "deletion":
            return TextAugmentation.random_deletion(tokens, p=p_del)
        elif aug_type == "swap":
            n_swaps = max(1, int(len(tokens) * p_swap))
            return TextAugmentation.random_swap(tokens, n=n_swaps)
        elif aug_type == "replacement" and vocab is not None:
            return TextAugmentation.random_replacement(tokens, vocab, p=p_repl)
        else:
            return tokens

class SlidingWindowDataset(Dataset):
    """Dataset for sliding window sampling of long documents."""
    
    def __init__(
        self,
        documents,
        tokenizer,
        block_size=128,
        stride=64,
        apply_augmentation=False,
        p_augment=0.5
    ):
        """
        Args:
            documents: List of documents (strings or token lists)
            tokenizer: Tokenizer for tokenizing text
            block_size: Size of text blocks to return
            stride: Stride for sliding window
            apply_augmentation: Whether to apply augmentation
            p_augment: Probability of applying augmentation
        """
        self.documents = documents
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.apply_augmentation = apply_augmentation
        self.p_augment = p_augment
        
        # Create examples
        self.examples = []
        self.create_examples()
        
    def create_examples(self):
        """Create examples using sliding window."""
        for doc_idx, document in enumerate(self.documents):
            # Tokenize if needed
            if isinstance(document, str):
                tokenized_doc = self.tokenizer.encode(document)
            else:
                tokenized_doc = document
                
            # Skip if document is too short
            if len(tokenized_doc) <= 2:  # Skip very short docs
                continue
                
            # Create windows
            for start_idx in range(0, len(tokenized_doc) - self.block_size + 1, self.stride):
                end_idx = start_idx + self.block_size
                
                # Extract block
                block = tokenized_doc[start_idx:end_idx]
                
                # Ensure block is right size
                if len(block) != self.block_size:
                    continue
                    
                self.examples.append({
                    "doc_idx": doc_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "input_ids": block
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example["input_ids"]
        
        # Apply augmentation with probability p_augment
        if self.apply_augmentation and random.random() < self.p_augment:
            # Get vocabulary for replacement
            vocab = list(range(self.tokenizer.vocab_size))
            
            # Apply augmentations
            input_ids = TextAugmentation.apply_augmentations(
                input_ids,
                vocab=vocab,
                p_del=0.05,
                p_swap=0.05,
                p_repl=0.05
            )
            
            # Ensure correct length
            if len(input_ids) < self.block_size:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
            elif len(input_ids) > self.block_size:
                input_ids = input_ids[:self.block_size]
            
        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()  # For language modeling
        }

class TemperatureBasedSampling:
    """Temperature-based sampling for more diverse training examples."""
    
    @staticmethod
    def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
        """Sample from logits with temperature, top-k, and nucleus sampling."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
            
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        # Apply nucleus (top-p) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_token

    @staticmethod
    def generate_diverse_samples(model, input_ids, num_samples=4, max_length=128, 
                                 temperature=1.2, top_k=50, top_p=0.95):
        """Generate diverse samples using temperature-based sampling."""
        model.eval()
        device = next(model.parameters()).device
        
        # Move input to device
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(device)
        batch_size = input_ids.size(0)
        
        # Expand for multiple samples
        input_ids = input_ids.repeat(num_samples, 1)
        
        # Track generated samples
        generated_samples = input_ids.clone()
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = model(input_ids=generated_samples)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Sample from the logits
                next_tokens = TemperatureBasedSampling.sample_from_logits(
                    next_token_logits, 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Add to generated samples
                next_tokens = next_tokens.unsqueeze(-1)
                generated_samples = torch.cat((generated_samples, next_tokens), dim=-1)
                
        # Reshape to [num_samples, batch_size, seq_len]
        generated_samples = generated_samples.view(num_samples, batch_size, -1)
        
        return generated_samples