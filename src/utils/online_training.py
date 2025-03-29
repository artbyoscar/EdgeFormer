#!/usr/bin/env python
# EdgeFormer - Simplified Online Training Pipeline
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

"""
This module implements a lightweight online training pipeline for EdgeFormer models.
It allows for on-device fine-tuning based on actual usage patterns with minimal
computational overhead.
"""

import os
import time
import logging
import threading
import queue
import json
from collections import deque
import numpy as np
import torch
from torch.optim import AdamW
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('online_training')

class OnlineTrainer:
    """
    Implements simplified on-device training for EdgeFormer models.
    
    Features:
    - Continuous learning from user interactions
    - Memory-efficient training with small batches
    - Prioritized experience replay
    - Automatic hyperparameter adaptation
    - Background training process
    """
    
    def __init__(self, model, tokenizer, config=None):
        """
        Initialize the online trainer.
        
        Args:
            model: EdgeFormer model to train
            tokenizer: Tokenizer for processing text
            config: Configuration dictionary with training parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set default configuration
        self.config = {
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'batch_size': 1,  # Start with single examples
            'max_batch_size': 4,  # Can grow to this size
            'min_samples_for_update': 5,  # Minimum samples before training
            'max_buffer_size': 1000,  # Maximum samples to keep
            'update_interval': 60,  # Seconds between updates
            'priority_alpha': 0.6,  # Priority sampling exponent
            'device': 'cpu',  # Default to CPU training
            'checkpoint_dir': 'checkpoints/online',  # Where to save checkpoints
            'checkpoint_interval': 3600,  # Seconds between checkpoints (1 hour)
            'dynamic_lr': True,  # Dynamically adjust learning rate
            'dynamic_batch': True,  # Dynamically adjust batch size
            'auto_save': True,  # Automatically save model
            'background_training': True,  # Train in background thread
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize training buffer (with priority)
        self.buffer = []
        self.priorities = []
        
        # Initialize recent loss tracking
        self.recent_losses = deque(maxlen=50)
        
        # Initialize statistics
        self.stats = {
            'updates': 0,
            'samples_seen': 0,
            'total_training_time': 0,
            'avg_loss': 0,
            'last_update_time': 0,
            'last_checkpoint_time': 0,
            'current_lr': self.config['learning_rate'],
            'current_batch_size': self.config['batch_size'],
        }
        
        # Set device
        self.device = self.config['device']
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move model to device if needed
        if self.model.device != self.device:
            self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Create checkpoint directory
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Initialize background training
        if self.config['background_training']:
            self.training_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.training_thread = threading.Thread(
                target=self._background_training_loop,
                daemon=True
            )
            self.training_thread.start()
            logger.info("Background training thread started")
    
    def add_sample(self, text, loss=None):
        """
        Add a training sample to the buffer.
        
        Args:
            text: Text to train on
            loss: Optional loss value for priority calculation
        """
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            logger.warning("Sample too short, skipping")
            return
        
        # Create sample
        sample = {
            'tokens': tokens,
            'text': text,
            'timestamp': time.time()
        }
        
        # Calculate priority
        if loss is not None:
            priority = loss
        else:
            # Use recent average loss if available, otherwise default
            priority = (sum(self.recent_losses) / len(self.recent_losses)) if self.recent_losses else 1.0
        
        # Add to buffer
        self.buffer.append(sample)
        self.priorities.append(priority)
        
        # Update statistics
        self.stats['samples_seen'] += 1
        
        # Check if buffer is too large
        if len(self.buffer) > self.config['max_buffer_size']:
            # Remove sample with lowest priority
            min_idx = np.argmin(self.priorities)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        
        # Check if we should do an update
        current_time = time.time()
        if (
            len(self.buffer) >= self.config['min_samples_for_update'] and
            current_time - self.stats['last_update_time'] >= self.config['update_interval']
        ):
            if self.config['background_training']:
                # Signal background thread
                self.training_queue.put(True)
            else:
                # Update immediately
                self.update()
    
    def _select_batch(self, batch_size):
        """
        Select a batch from the buffer using prioritized sampling.
        
        Args:
            batch_size: Number of samples to select
        
        Returns:
            List of samples
        """
        # Check if we have enough samples
        if len(self.buffer) == 0:
            return []
        
        # Adjust batch size if needed
        batch_size = min(batch_size, len(self.buffer))
        
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.config['priority_alpha']
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probs
        )
        
        # Get samples
        batch = [self.buffer[i] for i in indices]
        
        return batch
    
    def update(self):
        """
        Perform a training update using samples from the buffer.
        """
        # Check if we have enough samples
        if len(self.buffer) < self.config['min_samples_for_update']:
            logger.info(f"Not enough samples for update ({len(self.buffer)}/{self.config['min_samples_for_update']})")
            return
        
        # Record start time
        start_time = time.time()
        
        # Get batch size (dynamic if enabled)
        batch_size = self.stats['current_batch_size']
        
        # Select batch
        batch = self._select_batch(batch_size)
        if not batch:
            return
        
        # Prepare inputs
        max_length = max(len(sample['tokens']) for sample in batch)
        input_ids = []
        labels = []
        
        for sample in batch:
            tokens = sample['tokens']
            
            # Create input and label tensors (next token prediction)
            input_tensor = tokens[:-1]
            label_tensor = tokens[1:]
            
            # Pad if needed
            if len(input_tensor) < max_length - 1:
                input_tensor = input_tensor + [0] * (max_length - 1 - len(input_tensor))
                label_tensor = label_tensor + [-100] * (max_length - 1 - len(label_tensor))
            
            input_ids.append(input_tensor)
            labels.append(label_tensor)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Set to training mode
        self.model.train()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update recent losses
        self.recent_losses.append(loss.item())
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['avg_loss'] = sum(self.recent_losses) / len(self.recent_losses)
        self.stats['last_update_time'] = time.time()
        self.stats['total_training_time'] += time.time() - start_time
        
        # Dynamic learning rate adjustment
        if self.config['dynamic_lr'] and len(self.recent_losses) >= 10:
            self._adjust_learning_rate()
        
        # Dynamic batch size adjustment
        if self.config['dynamic_batch'] and self.stats['updates'] % 10 == 0:
            self._adjust_batch_size()
        
        # Checkpointing
        current_time = time.time()
        if (
            self.config['auto_save'] and
            current_time - self.stats['last_checkpoint_time'] >= self.config['checkpoint_interval']
        ):
            self.save_checkpoint()
        
        logger.info(f"Update completed: loss={loss.item():.4f}, samples={len(batch)}, lr={self.stats['current_lr']:.2e}")
    
    def _adjust_learning_rate(self):
        """
        Dynamically adjust learning rate based on recent losses.
        """
        # Calculate loss trend
        recent = list(self.recent_losses)
        if len(recent) < 10:
            return
        
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        # If loss is decreasing, keep current learning rate
        if second_half < first_half:
            return
        
        # If loss is increasing or flat, reduce learning rate
        current_lr = self.stats['current_lr']
        new_lr = current_lr * 0.8  # Reduce by 20%
        
        # Set minimum learning rate
        if new_lr < 1e-6:
            new_lr = 1e-6
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update statistics
        self.stats['current_lr'] = new_lr
        
        logger.info(f"Adjusted learning rate: {current_lr:.2e} -> {new_lr:.2e}")
    
    def _adjust_batch_size(self):
        """
        Dynamically adjust batch size based on training stability.
        """
        # If loss is stable, try increasing batch size
        if len(self.recent_losses) < 20:
            return
        
        # Calculate loss variance
        recent = list(self.recent_losses)
        variance = np.var(recent)
        mean = np.mean(recent)
        
        # Calculate coefficient of variation (normalized variance)
        cv = np.sqrt(variance) / mean if mean > 0 else float('inf')
        
        current_batch = self.stats['current_batch_size']
        
        # If training is stable (low variance), increase batch size
        if cv < 0.1 and current_batch < self.config['max_batch_size']:
            new_batch = min(current_batch + 1, self.config['max_batch_size'])
            self.stats['current_batch_size'] = new_batch
            logger.info(f"Increased batch size: {current_batch} -> {new_batch}")
        
        # If training is unstable (high variance), decrease batch size
        elif cv > 0.5 and current_batch > 1:
            new_batch = max(current_batch - 1, 1)
            self.stats['current_batch_size'] = new_batch
            logger.info(f"Decreased batch size: {current_batch} -> {new_batch}")
    
    def _background_training_loop(self):
        """
        Background thread for training updates.
        """
        logger.info("Background training thread started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for signal to train
                self.training_queue.get(timeout=5)
                
                # Perform update
                self.update()
                
                # Mark task as done
                self.training_queue.task_done()
            except queue.Empty:
                # No training needed, continue waiting
                pass
            except Exception as e:
                logger.error(f"Error in background training: {e}")
        
        logger.info("Background training thread stopped")
    
    def save_checkpoint(self):
        """
        Save model checkpoint and training state.
        """
        # Create timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create checkpoint directory
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / f"model_{timestamp}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save latest symlink
        latest_path = checkpoint_dir / "model_latest.pt"
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_path, latest_path)
        
        # Save training state
        state = {
            'timestamp': timestamp,
            'stats': self.stats,
            'config': self.config,
            'buffer_size': len(self.buffer),
            'optimizer_state': self.optimizer.state_dict()
        }
        
        state_path = checkpoint_dir / f"state_{timestamp}.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Update statistics
        self.stats['last_checkpoint_time'] = time.time()
        
        logger.info(f"Checkpoint saved to {model_path}")
    
    def load_checkpoint(self, path=None):
        """
        Load model checkpoint and training state.
        
        Args:
            path: Path to checkpoint directory or file
        """
        # If no path provided, use latest
        if path is None:
            checkpoint_dir = Path(self.config['checkpoint_dir'])
            path = checkpoint_dir / "model_latest.pt"
        
        path = Path(path)
        
        # Check if path exists
        if not path.exists():
            logger.error(f"Checkpoint path {path} does not exist")
            return False
        
        # Load model
        if path.is_file() and path.suffix == '.pt':
            # Load model state dict
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
            
            # Try to load state
            state_path = path.parent / f"state_{path.stem.split('_')[1]}.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                # Update stats and config
                self.stats.update(state['stats'])
                self.config.update(state['config'])
                
                # Update optimizer
                if 'optimizer_state' in state:
                    self.optimizer.load_state_dict(state['optimizer_state'])
                    # Move optimizer state to correct device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                
                logger.info(f"Training state loaded from {state_path}")
        
        return True
    
    def cleanup(self):
        """
        Clean up resources used by the trainer.
        """
        if self.config['background_training']:
            # Stop background thread
            self.stop_event.set()
            
            # Wait for thread to finish
            self.training_thread.join(timeout=5)
            
            logger.info("Background training thread stopped")
    
    def get_stats(self):
        """
        Get training statistics.
        
        Returns:
            Dictionary of training statistics
        """
        return {
            'updates': self.stats['updates'],
            'samples_seen': self.stats['samples_seen'],
            'buffer_size': len(self.buffer),
            'avg_loss': self.stats['avg_loss'],
            'current_lr': self.stats['current_lr'],
            'current_batch_size': self.stats['current_batch_size'],
            'total_training_time': self.stats['total_training_time'],
            'device': self.device
        }


# Convenience function to create a trainer
def create_trainer(model, tokenizer, config=None):
    """
    Create an online trainer for the model.
    
    Args:
        model: EdgeFormer model to train
        tokenizer: Tokenizer for processing text
        config: Optional configuration dictionary
    
    Returns:
        OnlineTrainer instance
    """
    return OnlineTrainer(model, tokenizer, config)