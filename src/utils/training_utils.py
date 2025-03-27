# src/utils/training_utils.py
import torch
import torch.nn as nn
import time
import logging
import math
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration for optimized training pipeline."""
    def __init__(
        self,
        learning_rate=5e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        mixed_precision=False,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=50,
        save_steps=500,
        eval_steps=100,
        output_dir="./output"
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.output_dir = output_dir

def get_optimizer_and_scheduler(model, config, num_training_steps):
    """Create optimizer and scheduler with appropriate settings."""
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.adam_epsilon
    )
    
    # Create learning rate scheduler
    if config.warmup_steps > 0:
        try:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=num_training_steps
            )
        except ImportError:
            # Simple linear warmup implementation if transformers not available
            def lr_lambda(current_step):
                if current_step < config.warmup_steps:
                    return float(current_step) / float(max(1, config.warmup_steps))
                return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - config.warmup_steps)))
            
            from torch.optim.lr_scheduler import LambdaLR
            scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    return optimizer, scheduler

def train(
    model,
    train_dataset,
    config,
    eval_dataset=None,
    collate_fn=None
):
    """Train the model with optimized pipeline including gradient accumulation and mixed precision."""
    # Set up CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup training dataloader with prefetch
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.per_device_train_batch_size,
        collate_fn=collate_fn,
        num_workers=4,  # Prefetch with multiple workers
        pin_memory=True  # Speed up data transfer to GPU
    )
    
    # Calculate total training steps
    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter, should be >= 1")
        
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    # Prepare optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Mixed precision training
    scaler = GradScaler() if config.mixed_precision else None
    
    # Track metrics
    tr_loss = 0.0
    logging_loss = 0.0
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Batch size per device = {config.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    logger.info(f"  Mixed precision training = {config.mixed_precision}")
    
    global_step = 0
    epochs_trained = 0
    model.zero_grad()
    
    train_iterator = range(epochs_trained, int(config.num_train_epochs))
    
    # Start training loop
    for epoch in train_iterator:
        epoch_start_time = time.time()
        epoch_iterator = train_dataloader
        
        for step, batch in enumerate(epoch_iterator):
            # Set model to training mode
            model.train()
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            
            # Forward pass with mixed precision if enabled
            if config.mixed_precision:
                with autocast():
                    outputs = model(**batch) if isinstance(batch, dict) else model(*batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    
                    # Scale loss for gradient accumulation
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps
                    
                    # Scale loss for backward
                    scaler.scale(loss).backward()
            else:
                outputs = model(**batch) if isinstance(batch, dict) else model(*batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Scale loss for gradient accumulation
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                
                loss.backward()
            
            tr_loss += loss.item()
            
            # Update weights after gradient accumulation steps
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.mixed_precision:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Update weights
                if config.mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
                
                model.zero_grad()
                global_step += 1
                
                # Logging
                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / config.logging_steps
                    logger.info(f"Step {global_step}: loss = {avg_loss:.4f}")
                    logging_loss = tr_loss
                
                # Evaluation
                if eval_dataset is not None and config.eval_steps > 0 and global_step % config.eval_steps == 0:
                    evaluate(model, eval_dataset, config, collate_fn)
                
                # Save model checkpoint
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    output_dir = f"{config.output_dir}/checkpoint-{global_step}"
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save model
                    model.save_pretrained(output_dir) if hasattr(model, 'save_pretrained') else torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                    
                    logger.info(f"Saving model checkpoint to {output_dir}")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config.num_train_epochs} completed in {epoch_time:.2f}s")
    
    return global_step, tr_loss / global_step

def evaluate(model, eval_dataset, config, collate_fn=None):
    """Evaluate the model."""
    # Set up CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    eval_batch_size = config.per_device_train_batch_size
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {eval_batch_size}")
    
    eval_loss = 0.0
    nb_eval_steps = 0
    
    model.eval()
    
    for batch in eval_dataloader:
        with torch.no_grad():
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            
            # Forward pass
            outputs = model(**batch) if isinstance(batch, dict) else model(*batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            eval_loss += loss.item()
        
        nb_eval_steps += 1
    
    # Calculate average loss
    eval_loss = eval_loss / nb_eval_steps
    
    logger.info(f"Evaluation loss: {eval_loss:.4f}")
    
    return {"loss": eval_loss}