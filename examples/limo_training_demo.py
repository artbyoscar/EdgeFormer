# examples/limo_training_demo.py
import argparse
import os
import logging
from src.training.limo_trainer import LIMOTrainer
from src.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging('limo_training')

def parse_args():
    parser = argparse.ArgumentParser(description="LIMO Training Demo")
    parser.add_argument('--input_dir', type=str, default='data/training_examples', 
                        help='Directory containing training examples')
    parser.add_argument('--output_dir', type=str, default='checkpoints/limo_trained',
                        help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, default='edgeformer-small',
                        help='Base model to train')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for training (cpu, cuda, mps)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode to add examples during training')
    parser.add_argument('--quality_threshold', type=float, default=0.75,
                        help='Quality threshold for example selection (0-1)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize LIMO trainer
    trainer = LIMOTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        quality_threshold=args.quality_threshold
    )
    
    # Load training examples
    logger.info(f"Loading training examples from {args.input_dir}")
    num_examples = trainer.load_examples(args.input_dir)
    logger.info(f"Loaded {num_examples} training examples")
    
    # Start training
    if args.interactive:
        logger.info("Starting interactive LIMO training")
        trainer.train_interactive(epochs=args.epochs)
    else:
        logger.info("Starting batch LIMO training")
        trainer.train(epochs=args.epochs)
    
    # Save the trained model
    trainer.save_model()
    logger.info(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()