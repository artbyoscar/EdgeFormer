#!/usr/bin/env python
# EdgeFormer - LIMO Training Script
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

"""
This script implements training for EdgeFormer models using the LIMO 
(Less Is More) approach, focusing on high-quality, curated examples
rather than massive datasets.

Usage:
python examples/train_limo.py --dataset data/limo_curated --model_size small --epochs 10 --output_dir checkpoints/limo_test
"""

import os
import sys
import argparse
import json
import time
import datetime
import logging
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.text_dataset import get_tokenizer, TextDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('limo_training')

class LIMOTrainer:
    """
    Implements the LIMO training approach for EdgeFormer models.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model
        self.initialize_model()
        
        # Log training parameters
        self.log_parameters()
    
    def log_parameters(self):
        """Log training parameters"""
        logger.info("Training parameters:")
        logger.info(f"  Model size: {self.args.model_size}")
        logger.info(f"  Epochs: {self.args.epochs}")
        logger.info(f"  Learning rate: {self.args.learning_rate}")
        logger.info(f"  Batch size: {self.args.batch_size}")
        logger.info(f"  Weight decay: {self.args.weight_decay}")
        logger.info(f"  Warmup steps: {self.args.warmup_steps}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dataset: {self.args.dataset}")
        logger.info(f"  Output directory: {self.args.output_dir}")
        
        # Log model configuration
        logger.info("Model configuration:")
        for key, value in self.config.__dict__.items():
            if not key.startswith('_'):
                logger.info(f"  {key}: {value}")
    
    def load_dataset(self):
        """Load and prepare the LIMO dataset"""
        logger.info(f"Loading dataset from {self.args.dataset}")
        
        if not os.path.exists(self.args.dataset):
            raise ValueError(f"Dataset path {self.args.dataset} does not exist")
        
        # Try to load the dataset
        dataset_path = Path(self.args.dataset)
        
        # Look for the tokenized dataset first
        tokenized_path = dataset_path / "limo_dataset_tokenized.json"
        json_path = dataset_path / "limo_dataset.json"
        txt_path = dataset_path / "limo_dataset.txt"
        
        # Check which format is available
        if tokenized_path.exists():
            logger.info("Found tokenized dataset")
            with open(tokenized_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract tokens
            self.dataset = [item['tokens'] for item in data]
            
        elif json_path.exists():
            logger.info("Found JSON dataset")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract text and tokenize
            self.dataset = []
            for item in tqdm(data, desc="Tokenizing"):
                tokens = self.tokenizer.encode(item['text'])
                self.dataset.append(tokens)
                
        elif txt_path.exists():
            logger.info("Found text dataset")
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by separator and tokenize
            samples = content.split('\n\n===\n\n')
            
            self.dataset = []
            for sample in tqdm(samples, desc="Tokenizing"):
                if sample.strip():
                    tokens = self.tokenizer.encode(sample)
                    self.dataset.append(tokens)
        else:
            raise ValueError(f"No dataset files found in {self.args.dataset}")
        
        logger.info(f"Loaded {len(self.dataset)} samples")
        
        # Split into training and validation sets (90/10 split)
        split_idx = int(0.9 * len(self.dataset))
        self.train_dataset = self.dataset[:split_idx]
        self.val_dataset = self.dataset[split_idx:]
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def initialize_model(self):
        """Initialize the EdgeFormer model"""
        logger.info(f"Initializing EdgeFormer model ({self.args.model_size})")
        
        # Define model configuration based on size
        if self.args.model_size == 'tiny':
            config = EdgeFormerConfig(
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=512,
                max_position_embeddings=1024
            )
        elif self.args.model_size == 'small':
            config = EdgeFormerConfig(
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=1024,
                max_position_embeddings=1024
            )
        elif self.args.model_size == 'medium':
            config = EdgeFormerConfig(
                hidden_size=512,
                num_hidden_layers=8,
                num_attention_heads=8,
                intermediate_size=2048,
                max_position_embeddings=1024
            )
        elif self.args.model_size == 'base':
            config = EdgeFormerConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=1024
            )
        else:
            raise ValueError(f"Invalid model size: {self.args.model_size}")
        
        # Add vocabulary size
        # Assuming tokenizer has a vocabulary
        config.vocab_size = len(self.tokenizer)
        
        # Add training-specific configuration
        config.attention_type = self.args.attention_type
        config.use_recurrent = self.args.use_recurrent
        config.use_budget = self.args.use_budget
        config.use_kv_cache = self.args.use_kv_cache
        
        # Save config
        self.config = config
        
        # Initialize model
        self.model = EdgeFormer(config)
        self.model.to(self.device)
        
        # Print model size
        model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Model size: {model_size:.2f}M parameters")
    
    def create_dataloader(self, dataset, batch_size, is_training=True):
        """Create dataloaders for training and validation"""
        
        class TextDataset(Dataset):
            def __init__(self, data, block_size=128):
                self.data = data
                self.block_size = block_size
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                tokens = self.data[idx]
                
                # Ensure tokens are not longer than block_size
                tokens = tokens[:self.block_size]
                
                # Create input and target tensors
                input_ids = torch.tensor(tokens[:-1] if len(tokens) > 1 else tokens, dtype=torch.long)
                labels = torch.tensor(tokens[1:] if len(tokens) > 1 else [-100], dtype=torch.long)
                
                return {
                    'input_ids': input_ids,
                    'labels': labels
                }
        
        # Create dataset
        dataset = TextDataset(dataset, block_size=self.args.block_size)
        
        # Create sampler
        sampler = RandomSampler(dataset) if is_training else None
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn
        )
        
        return dataloader
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences"""
        # Find max length in batch
        max_length = max(len(item['input_ids']) for item in batch)
        
        # Pad inputs and labels
        padded_inputs = []
        padded_labels = []
        
        for item in batch:
            input_ids = item['input_ids']
            labels = item['labels']
            
            # Calculate padding
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids with 0
            padded_input = torch.cat([
                input_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            
            # Pad labels with -100 (ignored in loss calculation)
            padded_label = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            
            padded_inputs.append(padded_input)
            padded_labels.append(padded_label)
        
        # Stack tensors
        inputs = torch.stack(padded_inputs)
        labels = torch.stack(padded_labels)
        
        return {
            'input_ids': inputs,
            'labels': labels
        }
    
    def get_optimizer_and_scheduler(self):
        """Set up optimizer and learning rate scheduler"""
        # Define optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Define scheduler
        if self.args.warmup_steps > 0:
            from transformers import get_linear_schedule_with_warmup
            
            # Calculate total training steps
            train_dataloader = self.create_dataloader(self.train_dataset, self.args.batch_size)
            total_steps = len(train_dataloader) * self.args.epochs
            
            # Create scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def train(self):
        """Train the model using the LIMO approach"""
        logger.info("Starting LIMO training")
        
        # Create dataloaders
        train_dataloader = self.create_dataloader(self.train_dataset, self.args.batch_size)
        val_dataloader = self.create_dataloader(self.val_dataset, self.args.batch_size, is_training=False)
        
        # Set up optimizer and scheduler
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        
        # Set up output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Training variables
        best_val_loss = float('inf')
        no_improvement_count = 0
        
        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            logger.info(f"Starting epoch {epoch}/{self.args.epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler)
            
            # Evaluate
            val_loss = self.evaluate(val_dataloader)
            
            # Log results
            logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                
                # Save model
                model_path = os.path.join(self.args.output_dir, f"model_epoch_{epoch}.pt")
                self.save_model(model_path)
                
                logger.info(f"New best model saved to {model_path}")
            else:
                no_improvement_count += 1
                
                # Early stopping
                if no_improvement_count >= self.args.patience:
                    logger.info(f"No improvement for {self.args.patience} epochs, stopping training")
                    break
        
        # Save final model
        final_model_path = os.path.join(self.args.output_dir, "final_model.pt")
        self.save_model(final_model_path)
        
        logger.info(f"Training completed, final model saved to {final_model_path}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        steps = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / steps
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        
        total_loss = 0
        steps = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Update metrics
                total_loss += loss.item()
                steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / steps
    
    def save_model(self, path):
        """Save model and configuration"""
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__
        }, path)
        
        # Save configuration separately
        config_path = os.path.join(os.path.dirname(path), "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='LIMO Training for EdgeFormer')
    
    # Dataset and output arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the LIMO dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'base'],
                        help='Model size')
    parser.add_argument('--attention_type', type=str, default='standard',
                        choices=['standard', 'mla', 'sliding_window', 'gqa'],
                        help='Attention mechanism type')
    parser.add_argument('--use_recurrent', action='store_true',
                        help='Use recurrent depth processing')
    parser.add_argument('--use_budget', action='store_true',
                        help='Use budget forcing')
    parser.add_argument('--use_kv_cache', action='store_true',
                        help='Use KV cache offloading')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm (0 to disable)')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LIMOTrainer(args)
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()