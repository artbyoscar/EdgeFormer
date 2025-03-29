#!/usr/bin/env python
# EdgeFormer - Simplified Online Training Demo
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

"""
This script demonstrates the simplified online training pipeline for EdgeFormer models.
It allows for interactive fine-tuning with minimal computational overhead.

Usage:
python examples/simplified_online_training_demo.py --model_path checkpoints/model.pt
"""

import os
import sys
import argparse
try:
    import readline
except ImportError:
    import pyreadline3 as readline
import time
import logging
from pathlib import Path

import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.text_dataset import get_tokenizer
from src.utils.online_training import create_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('online_training_demo')

class OnlineTrainingDemo:
    """
    Interactive demo for the simplified online training pipeline.
    """
    
    def __init__(self, args):
        self.args = args
        
        # Initialize components
        self.load_model()
        self.tokenizer = get_tokenizer()
        
        # Initialize trainer
        self.trainer_config = {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'update_interval': args.update_interval,
            'checkpoint_dir': args.output_dir,
            'device': args.device,
            'background_training': not args.no_background,
        }
        
        self.trainer = create_trainer(self.model, self.tokenizer, self.trainer_config)
        
        # Initialize history for interactive mode
        self.history = []
    
    def load_model(self):
        """Load pre-trained EdgeFormer model"""
        if self.args.model_path and os.path.exists(self.args.model_path):
            logger.info(f"Loading model from {self.args.model_path}")
            
            # Load model with config
            checkpoint = torch.load(self.args.model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                # Load from training checkpoint
                config_dict = checkpoint['config']
                if isinstance(config_dict, dict):
                    config = EdgeFormerConfig(**config_dict)
                else:
                    config = config_dict
                
                self.model = EdgeFormer(config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Load from regular checkpoint
                # Try to find config file
                config_path = os.path.join(os.path.dirname(self.args.model_path), "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    config = EdgeFormerConfig(**config_dict)
                else:
                    # Use default config
                    config = EdgeFormerConfig()
                
                self.model = EdgeFormer(config)
                self.model.load_state_dict(checkpoint)
        else:
            logger.info("No model path provided or model not found, initializing new model")
            
            # Initialize new model with custom config
            config = EdgeFormerConfig(
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=1024,
                max_position_embeddings=512,
                attention_type=self.args.attention_type
            )
            
            self.model = EdgeFormer(config)
        
        # Move model to device
        device = self.args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(device)
        
        # Print model size
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Model loaded: {param_count:.1f}M parameters")
    
    def generate_text(self, prompt, max_length=100):
        """Generate text from the model"""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.model.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_tensor,
                max_length=max_length,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p
            )[0].tolist()
        
        # Convert back to text
        output_text = self.tokenizer.decode(output_ids)
        
        return output_text
    
    def interactive_mode(self):
        """Run interactive console for online training demo"""
        print("\n===== EdgeFormer Online Training Demo =====")
        print("Commands:")
        print("  /help            - Show this help message")
        print("  /generate <text> - Generate text from the given prompt")
        print("  /train <text>    - Add text to training buffer")
        print("  /stats           - Show training statistics")
        print("  /save            - Save model checkpoint")
        print("  /exit            - Exit demo")
        print("Any other input will be treated as text for training.")
        print("=======================================\n")
        
        while True:
            try:
                # Get input
                user_input = input("> ").strip()
                
                # Process commands
                if user_input.startswith("/help"):
                    print("Commands:")
                    print("  /help            - Show this help message")
                    print("  /generate <text> - Generate text from the given prompt")
                    print("  /train <text>    - Add text to training buffer")
                    print("  /stats           - Show training statistics")
                    print("  /save            - Save model checkpoint")
                    print("  /exit            - Exit demo")
                
                elif user_input.startswith("/generate"):
                    # Extract prompt
                    prompt = user_input[10:].strip()
                    if not prompt:
                        print("Please provide a prompt for generation")
                        continue
                    
                    # Generate text
                    start_time = time.time()
                    output_text = self.generate_text(
                        prompt, 
                        max_length=self.args.max_length
                    )
                    elapsed = time.time() - start_time
                    
                    # Print generated text
                    print(f"\nGenerated ({elapsed:.2f}s):")
                    print("-" * 40)
                    print(output_text)
                    print("-" * 40)
                
                elif user_input.startswith("/train"):
                    # Extract text
                    text = user_input[7:].strip()
                    if not text:
                        print("Please provide text for training")
                        continue
                    
                    # Add to training buffer
                    self.trainer.add_sample(text)
                    print(f"Added to training buffer (size: {len(self.trainer.buffer)})")
                
                elif user_input.startswith("/stats"):
                    # Show training statistics
                    stats = self.trainer.get_stats()
                    print("\nTraining Statistics:")
                    print("-" * 40)
                    for key, value in stats.items():
                        if isinstance(value, float):
                            print(f"{key}: {value:.6f}")
                        else:
                            print(f"{key}: {value}")
                    print("-" * 40)
                
                elif user_input.startswith("/save"):
                    # Save checkpoint
                    self.trainer.save_checkpoint()
                    print("Model checkpoint saved")
                
                elif user_input.startswith("/exit"):
                    # Exit demo
                    print("Exiting demo...")
                    break
                
                else:
                    # Treat as text for training
                    if user_input:
                        self.trainer.add_sample(user_input)
                        print(f"Added to training buffer (size: {len(self.trainer.buffer)})")
            
            except KeyboardInterrupt:
                print("\nExiting demo...")
                break
            
            except Exception as e:
                print(f"Error: {e}")
        
        # Clean up
        self.trainer.cleanup()
    
    def batch_mode(self):
        """Run batch mode for automated online training demo"""
        # Load demo text
        if not self.args.input_file or not os.path.exists(self.args.input_file):
            logger.error(f"Input file {self.args.input_file} not found")
            return
        
        # Read input file
        with open(self.args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process lines
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            logger.info(f"Processing line {i+1}/{len(lines)}")
            
            # Add to training buffer
            self.trainer.add_sample(line)
            
            # Generate text occasionally
            if i % 10 == 0 and i > 0:
                output_text = self.generate_text(line[:50], max_length=100)
                logger.info(f"Generated: {output_text}")
        
        # Final stats
        stats = self.trainer.get_stats()
        logger.info("Training complete")
        logger.info(f"Stats: {stats}")
        
        # Save final model
        self.trainer.save_checkpoint()
        logger.info("Final model saved")
        
        # Clean up
        self.trainer.cleanup()


def main():
    parser = argparse.ArgumentParser(description='EdgeFormer Online Training Demo')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model')
    parser.add_argument('--attention_type', type=str, default='standard',
                        choices=['standard', 'mla', 'sliding_window', 'gqa'],
                        help='Attention type for new models')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for online training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Initial batch size for training')
    parser.add_argument('--update_interval', type=int, default=30,
                        help='Seconds between training updates')
    parser.add_argument('--output_dir', type=str, default='checkpoints/online',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for training and inference')
    parser.add_argument('--no_background', action='store_true',
                        help='Disable background training thread')
    
    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length for generated text')
    
    # Mode arguments
    parser.add_argument('--batch', action='store_true',
                        help='Run in batch mode instead of interactive mode')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file for batch mode')
    
    args = parser.parse_args()
    
    # Create demo
    demo = OnlineTrainingDemo(args)
    
    # Run demo
    if args.batch:
        demo.batch_mode()
    else:
        demo.interactive_mode()


if __name__ == "__main__":
    main()