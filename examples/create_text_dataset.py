# examples/create_text_dataset.py
import argparse
import logging
import os
import sys
import torch
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.text_dataset import TextDataset, create_wikitext_dataset, get_data_loaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('edgeformer')

def main():
    parser = argparse.ArgumentParser(description="Create and save a text dataset for EdgeFormer")
    parser.add_argument("--input_file", type=str, help="Path to input text file (optional)")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed dataset")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for training")
    parser.add_argument("--use_wikitext", action="store_true", help="Use WikiText-2 dataset instead of a custom file")
    parser.add_argument("--show_samples", action="store_true", help="Show sample sequences from the dataset")
    parser.add_argument("--sample_count", type=int, default=3, help="Number of samples to show")
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    if args.use_wikitext:
        try:
            dataset = create_wikitext_dataset(seq_length=args.seq_length)
            logger.info("Successfully created WikiText dataset")
        except Exception as e:
            logger.error(f"Error creating WikiText dataset: {e}")
            logger.info("Try installing the required library with: pip install datasets")
            return
    else:
        if not args.input_file:
            logger.error("Please provide --input_file or use --use_wikitext")
            return
            
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            return
            
        dataset = TextDataset(args.input_file, seq_length=args.seq_length)
        logger.info(f"Successfully created dataset from: {args.input_file}")
    
    # Save vocabulary
    if not hasattr(dataset, 'tokenizer') or dataset.tokenizer is None:
        vocab_file = os.path.join(args.output_dir, "vocab.pt")
        torch.save({
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char,
            'vocab_size': dataset.vocab_size
        }, vocab_file)
        logger.info(f"Saved vocabulary ({dataset.vocab_size} tokens) to {vocab_file}")
    
    # Show samples if requested
    if args.show_samples:
        logger.info(f"Showing {args.sample_count} sample sequences:")
        for i in range(min(args.sample_count, len(dataset))):
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            input_ids = sample["input_ids"]
            
            # Convert to readable text if we have a character mapping
            if hasattr(dataset, 'idx_to_char'):
                text = ''.join([dataset.idx_to_char[idx.item()] for idx in input_ids[:50]])
                logger.info(f"Sample {i+1}: {text}...")
            else:
                logger.info(f"Sample {i+1} input_ids: {input_ids[:20]}...")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(dataset, batch_size=4)
    logger.info(f"Created data loaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Save the dataset
    dataset_file = os.path.join(args.output_dir, "text_dataset.pt")
    torch.save(dataset.tokenized, dataset_file)
    logger.info(f"Saved tokenized dataset to {dataset_file}")
    
    logger.info("Dataset creation complete")

if __name__ == "__main__":
    main()