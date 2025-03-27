# examples/train_with_optimizations.py
import os
import sys
import logging
import torch
import argparse
from torch.utils.data import Dataset, random_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.training_utils import TrainingConfig, train
from src.utils.data_augmentation import SlidingWindowDataset, TextAugmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("training")

class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes."""
    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        
    def encode(self, text):
        """Simple character-level encoding for demonstration."""
        # Convert to ASCII values and limit to vocab size
        return [min(ord(c) % self.vocab_size, self.vocab_size - 1) for c in text]
    
    def decode(self, ids):
        """Decode ids back to text."""
        return ''.join(chr(id) for id in ids)

def load_text_data(data_dir, max_files=None):
    """Load text data from directory."""
    documents = []
    
    # List all text files
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    # Limit number of files if specified
    if max_files is not None:
        files = files[:max_files]
    
    # Load each file
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if text:
                    documents.append(text)
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                        if text:
                            documents.append(text)
                    break
                except:
                    continue
    
    return documents

def main(args):
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    documents = load_text_data(args.data_dir, max_files=args.max_files)
    
    if not documents:
        logger.warning("No documents found. Creating dummy data for testing.")
        # Create dummy data for testing
        documents = [
            "This is a sample text document for testing the EdgeFormer model.",
            "EdgeFormer uses Multi-Head Latent Attention to optimize performance.",
            "The model is designed to run efficiently on edge devices."
        ] * 10
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    
    # Create dataset with sliding window and augmentation
    dataset = SlidingWindowDataset(
        documents=documents,
        tokenizer=tokenizer,
        block_size=args.block_size,
        stride=args.stride,
        apply_augmentation=args.augmentation,
        p_augment=args.p_augment
    )
    
    # Split into train and eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create model
    config = EdgeFormerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.block_size,
        latent_size_factor=args.latent_size_factor
    )
    
    model = EdgeFormer(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    logger.info("Starting training")
    global_step, train_loss = train(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        eval_dataset=eval_dataset
    )
    
    logger.info(f"Training completed with {global_step} steps and final loss: {train_loss:.4f}")
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final-model")
    os.makedirs(final_output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_output_dir, "model.pt"))
    
    logger.info(f"Final model saved to {final_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EdgeFormer with optimizations")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing text files")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to load")
    parser.add_argument("--block_size", type=int, default=128, help="Size of text blocks")
    parser.add_argument("--stride", type=int, default=64, help="Stride for sliding window")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Intermediate size")
    parser.add_argument("--latent_size_factor", type=int, default=8, help="Latent size factor for MLA")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Augmentation arguments
    parser.add_argument("--augmentation", action="store_true", help="Apply data augmentation")
    parser.add_argument("--p_augment", type=float, default=0.5, help="Probability of applying augmentation")
    
    # Logging and saving arguments
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    main(args)