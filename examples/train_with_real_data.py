# examples/train_with_real_data.py
import argparse
import logging
import os
import sys
import time
import torch
from torch.optim import Adam
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.text_dataset import TextDataset, create_wikitext_dataset, get_data_loaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('edgeformer')

def train_model(model, train_loader, val_loader, epochs=5, learning_rate=1e-4, device="cpu", 
                checkpoint_dir="checkpoints", save_interval=1, test_generation=False):
    """
    Train the EdgeFormer model on a text dataset.
    
    Args:
        model: EdgeFormer model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ("cpu" or "cuda")
        checkpoint_dir: Directory to save checkpoints
        save_interval: Save checkpoints every N epochs
        test_generation: Whether to test text generation after training
    """
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_time = time.time() - train_start_time
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Valid]")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Update progress
                val_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_time = time.time() - val_start_time
        
        logger.info(f"Epoch {epoch}/{epochs} - "
                   f"Train Loss: {train_loss:.4f} ({train_time:.2f}s), "
                   f"Val Loss: {val_loss:.4f} ({val_time:.2f}s)")
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model.config
            }, checkpoint_path)
            logger.info(f"Saved best model with val_loss {val_loss:.4f} to {checkpoint_path}")
        
        # Save regular checkpoint at specified intervals
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model.config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': model.config
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")
    
    # Test text generation if requested
    if test_generation and hasattr(model, "generate"):
        logger.info("Testing text generation with trained model...")
        model.eval()
        
        # Generate text from a seed sequence
        seed_text = "EdgeFormer is a custom transformer that"
        
        # This assumes your model has a generate method
        # Modify this according to your model's generation method
        generated_text = model.generate(
            seed_text, 
            max_length=100,
            temperature=0.8,
            top_k=50
        )
        
        logger.info(f"Generated text:\n{generated_text}")
    
    return model

def main():
    logger.info("Starting EdgeFormer test...")
    parser = argparse.ArgumentParser(description="Train EdgeFormer with real text data")
    parser.add_argument("--dataset_file", type=str, help="Path to preprocessed dataset file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--use_wikitext", action="store_true", help="Use WikiText-2 dataset")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for the model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test_generation", action="store_true", help="Test text generation after training")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu or cuda)")
    parser.add_argument("--attention_type", type=str, default="standard", 
                      choices=["standard", "mla", "mla_window"], 
                      help="Attention mechanism to use")
    args = parser.parse_args()
    
    # Create dataset
    if args.use_wikitext:
        try:
            dataset = create_wikitext_dataset(seq_length=args.seq_length)
            logger.info("Successfully created WikiText dataset")
        except Exception as e:
            logger.error(f"Error creating WikiText dataset: {e}")
            logger.info("Try installing the required library with: pip install datasets")
            return
    elif args.dataset_file and os.path.exists(args.dataset_file):
        # Load preprocessed dataset
        tokenized_data = torch.load(args.dataset_file)
        logger.info(f"Loaded tokenized data with {len(tokenized_data)} tokens")
        
        # Also load vocabulary information if available
        vocab_file = os.path.join(os.path.dirname(args.dataset_file), "vocab.pt")
        if os.path.exists(vocab_file):
            vocab_info = torch.load(vocab_file)
            logger.info(f"Loaded vocabulary with {vocab_info['vocab_size']} tokens")
            
            # Create dataset with pre-tokenized data and vocabulary info
            dataset = TextDataset(tokenized_data, seq_length=args.seq_length)
            dataset.vocab_size = vocab_info['vocab_size']
            dataset.char_to_idx = vocab_info['char_to_idx']
            dataset.idx_to_char = vocab_info['idx_to_char']
        else:
            logger.warning("Vocabulary file not found, using default vocabulary size")
            dataset = TextDataset(tokenized_data, seq_length=args.seq_length)
            dataset.vocab_size = 256  # Default fallback
    else:
        logger.error("Please provide --dataset_file or use --use_wikitext")
        return
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        dataset, batch_size=args.batch_size
    )
    
    # Get vocabulary size from dataset
    vocab_size = dataset.vocab_size if hasattr(dataset, 'vocab_size') else 10000
    logger.info(f"Using vocabulary size: {vocab_size}")
    
    # Create model
    config = EdgeFormerConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        latent_size_factor=8,  # Adjust as needed
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.seq_length
    )
    
    model = EdgeFormer(config)
    if hasattr(model, 'set_attention_type'):
        model.set_attention_type(args.attention_type)
    elif hasattr(model, 'attention_type'):
        model.attention_type = args.attention_type
    logger.info(f"Created EdgeFormer model with {args.attention_type} attention")
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        test_generation=args.test_generation
    )
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()