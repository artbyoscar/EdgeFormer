# src/utils/training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import math
from datetime import datetime

logger = logging.getLogger("edgeformer")

class TextDataset(Dataset):
    """Dataset for text training data."""
    def __init__(self, file_path, tokenizer, max_length=512, stride=256):
        """
        Initialize dataset.
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            stride: Stride for sliding window tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Load data with multiple encoding fallbacks
        logger.info(f"Loading data from {file_path}")

        # Try multiple encodings
        encodings_to_try = ['utf-8', 'latin-1', 'utf-16', 'cp1252']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    self.text = f.read()
                logger.info(f"Successfully read file with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode file with encoding: {encoding}")
                if encoding == encodings_to_try[-1]:
                    raise ValueError(f"Could not read file {file_path} with any of the tried encodings: {encodings_to_try}")
                continue
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            
        # Tokenize text
        logger.info("Tokenizing text...")
        self.encodings = self.tokenize_text(self.text)

        # Calculate number of samples
        sample_count = len(self.encodings["input_ids"]) - 1
        self.n_samples = max(0, sample_count)  # Ensure we have at least 0 samples

        if self.n_samples == 0:
            logger.warning(f"No samples created from {file_path} - file may be too short for the given max_length and stride")
        else:
            logger.info(f"Created dataset with {self.n_samples} samples")
        
    
    def tokenize_text(self, text):
        """Tokenize text using sliding window approach."""
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=False,
            add_special_tokens=False
        )
        
        # Get input IDs and attention mask
        input_ids = tokenized["input_ids"][0]

        # If text is very short, just use it as is
        if len(input_ids) < self.max_length:
            logger.warning(f"Text is shorter than max_length ({len(input_ids)} < {self.max_length}), using as is")
            result = {
                "input_ids": [input_ids],
                "attention_mask": [torch.ones(len(input_ids))]
            }
            return result
        
        # Create samples with stride
        result = {"input_ids": [], "attention_mask": []}
        
        for i in range(0, len(input_ids), self.stride):
            # Get chunk
            chunk = input_ids[i:i + self.max_length]
            
            # Skip short chunks
            min_chunk_length = min(64, len(input_ids) // 2)  # More adaptive minimum length
            if len(chunk) < min_chunk_length:
                continue
            
            # Create attention mask
            attention_mask = torch.ones(len(chunk))
            
            # Add to result
            result["input_ids"].append(chunk)
            result["attention_mask"].append(attention_mask)

        # If no chunks were created, use the first max_length tokens
        if len(result["input_ids"]) == 0:
            chunk = input_ids[:self.max_length]
            attention_mask = torch.ones(len(chunk))
            result["input_ids"].append(chunk)
            result["attention_mask"].append(attention_mask)
        
        return result
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get input IDs and attention mask
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        
        # Get labels (next tokens)
        labels = self.encodings["input_ids"][idx + 1]
        
        # Ensure same length
        min_len = min(len(input_ids), len(labels))
        input_ids = input_ids[:min_len]
        attention_mask = attention_mask[:min_len]
        labels = labels[:min_len]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def create_dataloaders(train_file, val_file=None, tokenizer=None, batch_size=4, 
                       max_length=128, stride=64, val_split=0.1):
    """
    Create training and validation dataloaders.
    
    Args:
        train_file: Path to training file
        val_file: Path to validation file (optional)
        tokenizer: Tokenizer to use (if None, use default)
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for sliding window tokenization
        val_split: Validation split if val_file not provided
        
    Returns:
        train_dataloader, val_dataloader
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        logger.info("Creating default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create training dataset
    train_dataset = TextDataset(
        train_file,
        tokenizer,
        max_length=max_length,
        stride=stride
    )
    
    # Create validation dataset
    if val_file:
        val_dataset = TextDataset(
            val_file,
            tokenizer,
            max_length=max_length,
            stride=stride
        )
    elif val_split > 0 and train_dataset.n_samples > 0:
        # Split training dataset
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        # Use random_split
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size]
        )
    else:
        val_dataset = None
    
    # Check if datasets have samples
    if train_dataset.n_samples == 0:
        logger.warning("Training dataset has no samples, creating dummy data for testing")
        # Create dummy dataset with 10 random samples
        from torch.utils.data import TensorDataset
        dummy_inputs = torch.randint(0, tokenizer.vocab_size, (10, 16))
        dummy_masks = torch.ones_like(dummy_inputs)
        dummy_labels = torch.randint(0, tokenizer.vocab_size, (10, 16))
        train_dataset = TensorDataset(dummy_inputs, dummy_masks, dummy_labels)
        
        def collate_dummy_batch(batch):
            input_ids = torch.stack([item[0] for item in batch])
            attention_mask = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # Create training dataloader with dummy data
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_dummy_batch
        )
    else:
        # Create normal training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )
    
    # Similar check for validation dataset
    if val_dataset and hasattr(val_dataset, 'n_samples') and val_dataset.n_samples > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
    else:
        val_dataloader = None
    
    logger.info(f"Created dataloader with batch size {batch_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_dataloader, val_dataloader

def collate_batch(batch):
    """
    Collate batch for dataloader.
    
    Args:
        batch: Batch from dataset
        
    Returns:
        Collated batch
    """
    # Get batch elements
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Get maximum length
    max_len = max([len(ids) for ids in input_ids])
    
    # Pad inputs
    input_ids = [ids.tolist() + [0] * (max_len - len(ids)) for ids in input_ids]
    attention_mask = [mask.tolist() + [0] * (max_len - len(mask)) for mask in attention_mask]
    labels = [l.tolist() + [-100] * (max_len - len(l)) for l in labels]
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class GradualQuantizationScheduler:
    """
    Scheduler for gradual quantization during training.
    
    Gradually transitions from full precision to quantized weights during training
    to improve quality of quantized models.
    """
    def __init__(self, model, bits=4, group_size=128, symmetric=True, 
                 start_epoch=0, end_epoch=5, modules_to_exclude=None):
        """
        Initialize scheduler.
        
        Args:
            model: Model to quantize
            bits: Target bit width
            group_size: Group size for quantization
            symmetric: Whether to use symmetric quantization
            start_epoch: Epoch to start quantization
            end_epoch: Epoch to finish quantization
            modules_to_exclude: Modules to exclude from quantization
        """
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.modules_to_exclude = modules_to_exclude or []
        
        # Current quantization factor (0 = no quantization, 1 = full quantization)
        self.quantization_factor = 0.0
        
        # Store original parameters
        self.store_original_parameters()
        
        logger.info(f"Initialized gradual quantization scheduler: {bits}-bit, "
                    f"group_size={group_size}, symmetric={symmetric}, "
                    f"start_epoch={start_epoch}, end_epoch={end_epoch}")
    
    def store_original_parameters(self):
        """Store original parameters for gradual quantization."""
        self.original_params = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name not in self.modules_to_exclude:
                self.original_params[name] = module.weight.data.clone()
    
    def step(self, epoch):
        """
        Update quantization level based on current epoch.
        
        Args:
            epoch: Current epoch
        """
        # Calculate quantization factor
        if epoch < self.start_epoch:
            self.quantization_factor = 0.0
        elif epoch >= self.end_epoch:
            self.quantization_factor = 1.0
        else:
            # Linear increase
            self.quantization_factor = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        
        # Only update if factor changes
        if self.quantization_factor > 0:
            self.apply_gradual_quantization()
        
        logger.info(f"Epoch {epoch}: Quantization factor = {self.quantization_factor:.2f}")
    
    def apply_gradual_quantization(self):
        """Apply gradual quantization to model."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name not in self.modules_to_exclude:
                # Get original weights
                original_weight = self.original_params[name]
                
                # Get current weights (which might have been updated during training)
                current_weight = module.weight.data
                
                # Create temporary quantized weights
                from src.utils.weight_quantization import quantize_weight_only
                temp_module = nn.Linear(module.in_features, module.out_features, 
                                        bias=module.bias is not None)
                temp_module.weight.data = current_weight.clone()
                if module.bias is not None:
                    temp_module.bias.data = module.bias.data.clone()
                
                # Quantize temporary module
                quantized_module = quantize_weight_only(
                    temp_module,
                    bits=self.bits,
                    group_size=self.group_size,
                    symmetric=self.symmetric
                )
                
                # Dequantize for interpolation
                dequantized_weight = torch.zeros_like(current_weight)
                
                # Get dimensions
                output_dim, input_dim = current_weight.shape
                
                # Compute number of groups
                num_groups = (input_dim + self.group_size - 1) // self.group_size
                
                # Dequantize group by group
                for g in range(num_groups):
                    # Calculate start and end indices
                    start_idx = g * self.group_size
                    end_idx = min(start_idx + self.group_size, input_dim)
                    
                    if end_idx <= start_idx:
                        continue
                    
                    # Get weight slice
                    if self.symmetric:
                        # Dequantize symmetrically
                        weight_slice = quantized_module.weight[:, g, :end_idx-start_idx].float() * quantized_module.scales[:, g].unsqueeze(1)
                    else:
                        # Dequantize asymmetrically
                        weight_slice = (quantized_module.weight[:, g, :end_idx-start_idx].float() - quantized_module.zeros[:, g].unsqueeze(1)) * quantized_module.scales[:, g].unsqueeze(1)
                    
                    # Store dequantized weights
                    dequantized_weight[:, start_idx:end_idx] = weight_slice
                
                # Interpolate between current and quantized weights
                interpolated_weight = (1 - self.quantization_factor) * current_weight + self.quantization_factor * dequantized_weight
                
                # Update weights
                module.weight.data = interpolated_weight

def get_linear_warmup_cosine_decay_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        Scheduler function
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, train_dataloader, val_dataloader=None, 
                output_dir="checkpoints", num_epochs=3, 
                lr=5e-5, weight_decay=0.01, warmup_ratio=0.1,
                gradient_accumulation_steps=1, max_grad_norm=1.0,
                use_amp=False, log_interval=10, eval_interval=100, 
                save_interval=1000, device=None, enable_gradual_quantization=False,
                quantization_config=None):
    """
    Train or fine-tune the EdgeFormer model.
    
    Args:
        model: EdgeFormer model to train
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader (optional)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Ratio of warmup steps
        gradient_accumulation_steps: Number of steps for gradient accumulation
        max_grad_norm: Maximum gradient norm
        use_amp: Whether to use automatic mixed precision
        log_interval: Log interval in steps
        eval_interval: Evaluation interval in steps
        save_interval: Save interval in steps
        device: Device to use (default: auto-detect)
        enable_gradual_quantization: Whether to use gradual quantization
        quantization_config: Configuration for gradual quantization
        
    Returns:
        Trained model and training stats
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training on device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Calculate total number of training steps
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    
    # Create scheduler
    num_warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_warmup_cosine_decay_scheduler(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Initialize gradual quantization if enabled
    quantization_scheduler = None
    if enable_gradual_quantization:
        if quantization_config is None:
            quantization_config = {
                "bits": 4,
                "group_size": 128,
                "symmetric": True,
                "start_epoch": num_epochs // 2,  # Start halfway through training
                "end_epoch": num_epochs - 1      # End at the last epoch
            }
        
        from src.utils.weight_quantization import weight_only_quantize_model
        
        # Initialize scheduler
        quantization_scheduler = GradualQuantizationScheduler(
            model, 
            **quantization_config
        )
    
    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Initialize training stats
    training_stats = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "steps": [],
        "epochs": [],
        "best_val_loss": float("inf"),
        "best_step": 0
    }
    
    # Training loop
    global_step = 0
    train_loss = 0.0
    model.train()
    
    logger.info(f"Starting training for {num_epochs} epochs, {total_steps} steps")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Update quantization scheduler
        if quantization_scheduler:
            quantization_scheduler.step(epoch)
        
        # Initialize progress bar
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with or without mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs["loss"]
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs["loss"]
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            # Update training stats
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
            progress_bar.update(1)
            
            # Check if we should update
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                if use_amp:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update parameters
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Increment global step
                global_step += 1
                
                # Log training progress
                if global_step % log_interval == 0:
                    # Calculate average loss
                    avg_loss = train_loss / log_interval
                    train_loss = 0.0
                    
                    # Get current learning rate
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Add to training stats
                    training_stats["train_loss"].append(avg_loss)
                    training_stats["learning_rates"].append(current_lr)
                    training_stats["steps"].append(global_step)
                    training_stats["epochs"].append(epoch + step / len(train_dataloader))
                    
                    logger.info(f"Step {global_step}: loss = {avg_loss:.4f}, lr = {current_lr:.6f}")
                
                # Evaluate model
                if val_dataloader is not None and global_step % eval_interval == 0:
                    # Evaluate
                    val_loss = evaluate_model(model, val_dataloader, device, use_amp)
                    
                    # Add to training stats
                    training_stats["val_loss"].append(val_loss)
                    
                    # Check if best model
                    if val_loss < training_stats["best_val_loss"]:
                        training_stats["best_val_loss"] = val_loss
                        training_stats["best_step"] = global_step
                        
                        # Save best model
                        save_checkpoint(
                            model, 
                            optimizer, 
                            scheduler, 
                            training_stats,
                            os.path.join(run_dir, "best_model"),
                            global_step,
                            scaler=scaler
                        )
                        
                        logger.info(f"New best model: val_loss = {val_loss:.4f}")
                    
                    # Back to training mode
                    model.train()
                
                # Save checkpoint
                if global_step % save_interval == 0:
                    save_checkpoint(
                        model, 
                        optimizer, 
                        scheduler, 
                        training_stats,
                        os.path.join(run_dir, f"checkpoint-{global_step}"),
                        global_step,
                        scaler=scaler
                    )
        
        # End of epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")
        
        # Save epoch checkpoint
        save_checkpoint(
            model, 
            optimizer, 
            scheduler, 
            training_stats,
            os.path.join(run_dir, f"epoch-{epoch+1}"),
            global_step,
            scaler=scaler
        )
        
        # Evaluate at the end of each epoch
        if val_dataloader is not None:
            val_loss = evaluate_model(model, val_dataloader, device, use_amp)
            logger.info(f"Epoch {epoch+1} validation: loss = {val_loss:.4f}")
            
            # Add to training stats
            training_stats["val_loss"].append(val_loss)
            
            # Check if best model
            if val_loss < training_stats["best_val_loss"]:
                training_stats["best_val_loss"] = val_loss
                training_stats["best_step"] = global_step
                
                # Save best model
                save_checkpoint(
                    model, 
                    optimizer, 
                    scheduler, 
                    training_stats,
                    os.path.join(run_dir, "best_model"),
                    global_step,
                    scaler=scaler
                )
                
                logger.info(f"New best model: val_loss = {val_loss:.4f}")
            
            # Back to training mode
            model.train()
    
    # End of training
    logger.info(f"Training completed: {num_epochs} epochs, {global_step} steps")
    
    # Save final model
    save_checkpoint(
        model, 
        optimizer, 
        scheduler, 
        training_stats,
        os.path.join(run_dir, "final_model"),
        global_step,
        scaler=scaler
    )
    
    # Plot training stats
    plot_training_stats(training_stats, os.path.join(run_dir, "training_stats.png"))
    
    return model, training_stats, run_dir

def evaluate_model(model, dataloader, device, use_amp=False):
    """
    Evaluate model on dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for evaluation
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with or without mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
            
            # Get loss
            loss = outputs["loss"]
            
            # Update total loss
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, training_stats, output_dir, 
                   global_step, scaler=None):
    """
    Save checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        training_stats: Training statistics
        output_dir: Output directory
        global_step: Global step
        scaler: Gradient scaler for mixed precision
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Save optimizer and scheduler
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "global_step": global_step,
        "scaler": scaler.state_dict() if scaler else None
    }, os.path.join(output_dir, "optimizer.pt"))
    
    # Save training stats
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f)
    
    logger.info(f"Saved checkpoint to {output_dir}")

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_dir=None, 
                   device=None, scaler=None):
    """
    Load checkpoint.
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load state
        scheduler: Scheduler to load state
        checkpoint_dir: Directory containing checkpoint
        device: Device to load checkpoint to
        scaler: Gradient scaler for mixed precision
        
    Returns:
        model, optimizer, scheduler, training_stats, global_step, scaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model state
    model_path = os.path.join(checkpoint_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load optimizer and scheduler
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    checkpoint = torch.load(optimizer_path, map_location=device)
    
    global_step = checkpoint["global_step"]
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    if scaler and "scaler" in checkpoint and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    
    # Load training stats
    training_stats_path = os.path.join(checkpoint_dir, "training_stats.json")
    with open(training_stats_path, "r") as f:
        training_stats = json.load(f)
    
    logger.info(f"Loaded checkpoint from {checkpoint_dir} at step {global_step}")
    
    return model, optimizer, scheduler, training_stats, global_step, scaler

def plot_training_stats(training_stats, output_path=None):
    """
    Plot training statistics.
    
    Args:
        training_stats: Training statistics
        output_path: Path to save plot
    """
    plt.figure(figsize=(15, 12))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(training_stats["steps"], training_stats["train_loss"], label="Training")
    
    if training_stats["val_loss"]:
        # Create x-axis points for validation loss (may be fewer points)
        val_steps = []
        for i, step in enumerate(training_stats["steps"]):
            if i % (len(training_stats["steps"]) // len(training_stats["val_loss"])) == 0 and len(val_steps) < len(training_stats["val_loss"]):
                val_steps.append(step)
        
        if len(val_steps) < len(training_stats["val_loss"]):
            val_steps = training_stats["steps"][-len(training_stats["val_loss"]):]
        
        plt.plot(val_steps, training_stats["val_loss"], label="Validation")
    
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 1, 2)
    plt.plot(training_stats["steps"], training_stats["learning_rates"])
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved training stats plot to {output_path}")
    
    plt.close