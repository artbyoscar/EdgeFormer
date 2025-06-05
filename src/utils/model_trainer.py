# src/utils/model_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import os
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('edgeformer')

class EdgeFormerTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        val_dataloader=None, 
        lr=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints"
    ):
        """
        Initialize the EdgeFormer trainer.
        
        Args:
            model: The EdgeFormer model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            lr: Learning rate
            weight_decay: Weight decay for AdamW optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_grad_norm: Maximum gradient norm for gradient clipping
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Set up loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Set up learning rate scheduler
        # Simple linear scheduler with warmup
        self.lr_scheduler = self._create_lr_scheduler(warmup_steps)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_lr_scheduler(self, warmup_steps):
        """Create a learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Linear decay
            return max(0.0, 1.0 - float(current_step - warmup_steps) / float(max(1, 100000 - warmup_steps)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self, epochs=1, eval_steps=1000, save_steps=5000):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            eval_steps: Evaluate model every eval_steps steps
            save_steps: Save model every save_steps steps
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Training statistics
        stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        # Start timing
        total_start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Set model to training mode
            self.model.train()
            
            # Training for one epoch
            epoch_loss = 0
            num_batches = len(self.train_dataloader)
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    input_ids = batch
                    labels = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                    # Handle case where DataLoader returns a single item
                    input_ids = batch[0]
                    labels = batch[0]
                else:
                    # Assuming batch is a dictionary
                    input_ids = batch['input_ids']
                    labels = batch.get('labels', input_ids)
                
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Calculate loss
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = self.loss_fn(shift_logits.view(-1, self.model.config.vocab_size), 
                                   shift_labels.view(-1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # Update stats
                epoch_loss += loss.item()
                stats['train_losses'].append(loss.item())
                stats['learning_rates'].append(self.lr_scheduler.get_last_lr()[0])
                
                self.global_step += 1
                
                # Evaluation
                if self.val_dataloader is not None and self.global_step % eval_steps == 0:
                    val_loss = self.evaluate()
                    stats['val_losses'].append(val_loss)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_model("best_model.pt")
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_model(f"checkpoint_{self.global_step}.pt")
            
            # End of epoch
            epoch_loss /= num_batches
            epoch_end_time = time.time()
            logger.info(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f}s, "
                       f"loss: {epoch_loss:.4f}")
            
            # Save model at the end of each epoch
            self.save_model(f"epoch_{epoch+1}.pt")
        
        # End of training
        total_end_time = time.time()
        logger.info(f"Training completed in {total_end_time - total_start_time:.2f}s")
        
        # Save final model
        self.save_model("final_model.pt")
        
        return stats
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        logger.info("Evaluating model")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation loop
        val_loss = 0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    input_ids = batch
                    labels = batch
                elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                    # Handle case where DataLoader returns a single item
                    input_ids = batch[0]
                    labels = batch[0]
                else:
                    # Assuming batch is a dictionary
                    input_ids = batch['input_ids']
                    labels = batch.get('labels', input_ids)
                
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Calculate loss
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = self.loss_fn(shift_logits.view(-1, self.model.config.vocab_size), 
                                   shift_labels.view(-1))
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= num_batches
        
        # Set model back to training mode
        self.model.train()
        
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        return val_loss
    
    def save_model(self, filename):
        """
        Save model to checkpoint directory.
        
        Args:
            filename: Filename for the checkpoint
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        logger.info(f"Loading model from {filepath}")
        
        checkpoint = torch.load(filepath)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Model loaded successfully. Global step: {self.global_step}")
    
    def generate_text(self, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt to start generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Tokenize prompt (simple character-level tokenization for demo)
        if isinstance(prompt, str):
            input_ids = torch.tensor([[ord(c) % self.model.config.vocab_size for c in prompt]])
        else:
            input_ids = prompt
            
        # Generate text
        generated_text = prompt if isinstance(prompt, str) else ""
        past_key_values = None
        
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :].squeeze() / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = torch.topk(next_token_logits, k=top_k)[1]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to input for next step
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Convert to character and add to generated text (simple character-level)
                if isinstance(prompt, str):
                    next_char = chr(next_token.item() % 128)  # Ensure valid ASCII
                    generated_text += next_char
        
        return generated_text

def load_pretrained_weights(model, weights_path, adapter_config=None):
    """
    Load pre-trained weights into EdgeFormer model.
    
    Args:
        model: The EdgeFormer model
        weights_path: Path to pre-trained weights
        adapter_config: Optional configuration for weight adaptation
        
    Returns:
        Model with loaded weights
    """
    logger = logging.getLogger('edgeformer')
    logger.info(f"Loading pre-trained weights from {weights_path}")
    
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        return model
    
    try:
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Our own checkpoint format
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and all(k.startswith('model.') for k in checkpoint.keys()):
            # HuggingFace format with 'model.' prefix
            state_dict = {k[6:]: v for k, v in checkpoint.items() if k.startswith('model.')}
        else:
            # Assume direct state dict
            state_dict = checkpoint
        
        # Apply adapter if provided
        if adapter_config is not None:
            state_dict = adapt_weights(state_dict, model.state_dict(), adapter_config)
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info("Pre-trained weights loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading pre-trained weights: {e}")
    
    return model

def adapt_weights(source_state_dict, target_state_dict, adapter_config):
    """
    Adapt weights from source format to target format.
    
    Args:
        source_state_dict: Source state dict
        target_state_dict: Target state dict
        adapter_config: Adapter configuration
        
    Returns:
        Adapted state dict
    """
    logger = logging.getLogger('edgeformer')
    logger.info("Adapting weights between different model formats")
    
    adapted_state_dict = {}
    
    # Map of source keys to target keys
    key_mapping = adapter_config.get('key_mapping', {})
    
    # Process each key in the source state dict
    for source_key, param in source_state_dict.items():
        # Skip keys that should be excluded
        if source_key in adapter_config.get('exclude_keys', []):
            continue
        
        # Get target key from mapping or use source key
        target_key = key_mapping.get(source_key, source_key)
        
        # Skip if target key doesn't exist in target state dict
        if target_key not in target_state_dict:
            logger.debug(f"Skipping key {source_key} -> {target_key} (not in target model)")
            continue
        
        # Get target param shape
        target_shape = target_state_dict[target_key].shape
        
        # Handle parameter resizing if shapes don't match
        if param.shape != target_shape:
            if adapter_config.get('allow_shape_change', False):
                # Try to reshape or adapt tensor dimensions
                try:
                    # Handle dimension change
                    if param.dim() != target_state_dict[target_key].dim():
                        logger.warning(f"Dimension mismatch for {target_key}: {param.shape} vs {target_shape}")
                        continue
                    
                    # Handle simple cases like vocabulary size change
                    if len(param.shape) == 2 and param.shape[0] != target_shape[0]:
                        # For embedding matrices, take the min size
                        min_size = min(param.shape[0], target_shape[0])
                        param = param[:min_size, :]
                    
                    if len(param.shape) == 2 and param.shape[1] != target_shape[1]:
                        min_size = min(param.shape[1], target_shape[1])
                        param = param[:, :min_size]
                    
                    logger.info(f"Resized parameter {target_key}: {param.shape} -> {target_shape}")
                except Exception as e:
                    logger.warning(f"Failed to resize parameter {target_key}: {e}")
                    continue
            else:
                logger.warning(f"Shape mismatch for {target_key}: {param.shape} vs {target_shape}")
                continue
        
        # Add to adapted state dict
        adapted_state_dict[target_key] = param
    
    logger.info(f"Adapted {len(adapted_state_dict)}/{len(source_state_dict)} parameters")
    
    return adapted_state_dict

def create_training_script():
    """Generate a complete training script."""
    return """
# examples/train_model.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.model_trainer import EdgeFormerTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('edgeformer')

def main():
    parser = argparse.ArgumentParser(description="Train EdgeFormer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()
    
    # Create model
    config = EdgeFormerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
        intermediate_size=1024,
        max_position_embeddings=1024,
        vocab_size=32000,
    )
    
    model = EdgeFormer(config)
    
    # Create a simple synthetic dataset for demonstration
    # In a real scenario, you'd use actual text data
    logger.info("Creating synthetic dataset...")
    num_samples = 1000
    
    # Generate random sequences for training
    train_data = torch.randint(0, config.vocab_size, (num_samples, args.seq_length))
    
    # Create a smaller validation set
    val_data = torch.randint(0, config.vocab_size, (100, args.seq_length))
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create trainer
    trainer = EdgeFormerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    logger.info(f"Training for {args.epochs} epochs...")
    stats = trainer.train(epochs=args.epochs, eval_steps=50, save_steps=200)
    
    logger.info("Training completed!")
    
    # Test the trained model on a simple prompt
    model.eval()
    prompt = "EdgeFormer is a custom transformer that"
    
    logger.info(f"Generating text from prompt: '{prompt}'")
    generated_text = trainer.generate_text(prompt, max_length=100, temperature=0.7)
    
    logger.info(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
"""
