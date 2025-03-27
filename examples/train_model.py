# examples/train_model.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import argparse
import logging
import time

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
    parser.add_argument("--test_generation", action="store_true", help="Test text generation after training")
    parser.add_argument("--attention_type", type=str, default="standard", 
                       choices=["standard", "mla", "mla_window"], 
                       help="Attention mechanism to use")
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
    
    # Set attention type if the model has that attribute
    if hasattr(model, 'attention_type'):
        model.attention_type = args.attention_type
        logger.info(f"Set attention type to {args.attention_type}")
    elif hasattr(model, 'set_attention_type'):
        model.set_attention_type(args.attention_type)
        logger.info(f"Set attention type to {args.attention_type}")
    else:
        logger.warning(f"Model doesn't have attention_type attribute. Using default attention.")
    
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
    start_time = time.time()
    stats = trainer.train(epochs=args.epochs, eval_steps=50, save_steps=200)
    end_time = time.time()
    
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds!")
    
    # Test the trained model on a simple prompt
    if args.test_generation:
        model.eval()
        prompt = "EdgeFormer is a custom transformer that"
        
        logger.info(f"Generating text from prompt: '{prompt}'")
        generated_text = trainer.generate_text(prompt, max_length=100, temperature=0.7)
        
        logger.info(f"Generated text: {generated_text}")
        
        # Test with different temperatures
        logger.info("Testing generation with different temperatures:")
        
        for temp in [0.5, 1.0, 1.5]:
            logger.info(f"Temperature: {temp}")
            gen_text = trainer.generate_text(prompt, max_length=50, temperature=temp)
            logger.info(f"Generated: {gen_text}")

def create_text_dataset_from_file(file_path, tokenizer, seq_length):
    """
    Create a dataset from a text file.
    
    Args:
        file_path: Path to text file
        tokenizer: Tokenizer function or object
        seq_length: Sequence length
        
    Returns:
        Dataset of tokenized sequences
    """
    logger.info(f"Creating dataset from {file_path}")
    
    # Read text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize text
    if hasattr(tokenizer, 'encode'):
        # Use tokenizer object
        tokens = tokenizer.encode(text)
    else:
        # Use tokenizer function
        tokens = tokenizer(text)
    
    # Create sequences
    sequences = []
    for i in range(0, len(tokens) - seq_length, seq_length // 2):
        sequences.append(tokens[i:i + seq_length])
    
    # Convert to tensor
    sequences_tensor = torch.tensor(sequences)
    
    return TensorDataset(sequences_tensor)

if __name__ == "__main__":
    main()