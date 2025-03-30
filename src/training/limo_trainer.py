# src/training/limo_trainer.py
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.utils.logging_utils import get_logger
from src.utils.device_utils import get_optimal_device
from src.models.model_optimizer import optimize_model_for_device

logger = get_logger('limo_trainer')

class LIMODataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the input and output
        inputs = self.tokenizer(example['input'], return_tensors='pt', padding='max_length', 
                               truncation=True, max_length=512)
        outputs = self.tokenizer(example['output'], return_tensors='pt', padding='max_length',
                                truncation=True, max_length=512)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': outputs['input_ids'].squeeze(),
            'quality_score': example.get('quality_score', 1.0)
        }

class LIMOTrainer:
    def __init__(self, model_name, output_dir, batch_size=4, learning_rate=5e-5, 
                 device=None, quality_threshold=0.75):
        self.model_name = model_name
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.quality_threshold = quality_threshold
        self.examples = []
        
        # Set device
        self.device = device if device else get_optimal_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        try:
            logger.info(f"Loading model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Apply device-specific optimizations
            from src.models.model_optimizer import optimize_model_for_device
            self.model = optimize_model_for_device(self.model, self.model.config)
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_examples(self, input_dir):
        """Load training examples from files in the input directory"""
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory {input_dir} does not exist")
            return 0
            
        example_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        
        for file_name in example_files:
            file_path = os.path.join(input_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Split the content into input and output
                parts = content.split('---OUTPUT---')
                if len(parts) == 2:
                    input_text = parts[0].strip()
                    output_text = parts[1].strip()
                    
                    # Calculate quality score using LIMO metrics
                    quality_score = self._calculate_quality_score(input_text, output_text)
                    
                    if quality_score >= self.quality_threshold:
                        self.examples.append({
                            'input': input_text,
                            'output': output_text,
                            'quality_score': quality_score
                        })
                        logger.debug(f"Added example from {file_name} with quality score {quality_score:.2f}")
                    else:
                        logger.debug(f"Skipped example from {file_name} with low quality score {quality_score:.2f}")
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                
        return len(self.examples)
    
    def _calculate_quality_score(self, input_text, output_text):
        """Calculate a quality score for the example using LIMO metrics"""
        # This is a placeholder for the actual LIMO metrics
        # In a real implementation, you would use more sophisticated metrics
        
        # For now, use a simple heuristic based on:
        # - Length ratio between input and output
        # - Text complexity
        import textstat
        
        # Length metrics
        input_words = len(input_text.split())
        output_words = len(output_text.split())
        
        # Avoid division by zero
        if input_words == 0:
            return 0.0
            
        # Word ratio score (penalize if output is too short or too long)
        ratio = output_words / input_words
        if ratio < 0.3:
            ratio_score = ratio  # Too short
        elif ratio > 3.0:
            ratio_score = 3.0 / ratio  # Too long
        else:
            ratio_score = 1.0  # Good ratio
            
        # Complexity score
        input_complexity = textstat.flesch_reading_ease(input_text)
        output_complexity = textstat.flesch_reading_ease(output_text)
        
        # Normalize complexity scores to 0-1 range
        input_complexity = max(0, min(100, input_complexity)) / 100
        output_complexity = max(0, min(100, output_complexity)) / 100
        
        # Complexity difference score (penalize if output is much simpler or more complex)
        complexity_diff = abs(output_complexity - input_complexity)
        complexity_score = 1.0 - complexity_diff
        
        # Combine scores
        quality_score = 0.5 * ratio_score + 0.5 * complexity_score
        
        return quality_score
    
    def train(self, epochs=3):
        """Train the model for a specified number of epochs"""
        if not self.examples:
            logger.warning("No training examples available")
            return
            
        # Create dataset and dataloader
        dataset = LIMODataset(self.examples, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Weight examples by quality score
                quality_scores = batch['quality_score'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Apply quality weighting
                weighted_loss = loss * quality_scores.mean()
                
                # Backward pass
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Log average loss for the epoch
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            self._save_checkpoint(epoch + 1)
    
    def train_interactive(self, epochs=3):
        """Train the model in interactive mode, allowing user to add examples"""
        # Implement interactive training here
        pass
    
    def _save_checkpoint(self, epoch):
        """Save a checkpoint of the model"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def save_model(self):
        """Save the final trained model"""
        # Save model and tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Saved model to {self.output_dir}")