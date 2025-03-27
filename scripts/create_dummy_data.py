"""
Create dummy training data for EdgeFormer.

This script generates synthetic text data that can be used for testing
the EdgeFormer training pipeline when real data is not available.
"""

import argparse
import os
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dummy-data")

# Sample sentences for generating dummy data
SAMPLE_SENTENCES = [
    "EdgeFormer is a custom transformer implementation for edge devices.",
    "Multi-Head Latent Attention reduces the KV cache size significantly.",
    "Grouped-Query Attention allows query heads to share key and value heads.",
    "Sparse MLP implementation uses sparsity masks to reduce computation.",
    "Sliding Window Attention efficiently handles longer sequences.",
    "INT4 quantization achieves up to 8x memory reduction.",
    "Weight-Only Quantization further reduces model size.",
    "KV Cache Offloading supports processing very long sequences.",
    "RDNA3 Optimizations target AMD Radeon graphics cards.",
    "DirectML Acceleration provides GPU support for AMD graphics.",
    "The training pipeline includes gradual quantization support.",
    "Data augmentation techniques improve training robustness.",
    "The model achieves significant memory efficiency improvements.",
    "Benchmark results show promising performance on edge devices.",
    "The project structure includes examples and utility scripts.",
    "Memory tracking helps optimize the model for low-resource environments.",
    "Text generation demos showcase the model's capabilities.",
    "Tokenization is an important part of the text processing pipeline.",
    "Model export functionality allows deployment to different platforms.",
    "Documentation is essential for understanding the codebase.",
]

def generate_dummy_document(num_paragraphs, sentences_per_paragraph):
    """
    Generate a dummy document with the specified structure.
    
    Args:
        num_paragraphs: Number of paragraphs to generate
        sentences_per_paragraph: Number of sentences per paragraph
        
    Returns:
        Generated text document
    """
    document = []
    
    for _ in range(num_paragraphs):
        paragraph = []
        for _ in range(sentences_per_paragraph):
            paragraph.append(random.choice(SAMPLE_SENTENCES))
        document.append(" ".join(paragraph))
    
    return "\n\n".join(document)

def main(args):
    """
    Main function to generate and save dummy data.
    
    Args:
        args: Command line arguments
    """
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Determine number of documents based on split
        if split == "train":
            num_docs = args.num_train_docs
        elif split == "valid":
            num_docs = args.num_valid_docs
        else:  # test
            num_docs = args.num_test_docs
        
        logger.info(f"Generating {num_docs} {split} documents...")
        
        for i in range(num_docs):
            # Generate document with random length
            num_paragraphs = random.randint(5, 20)
            sentences_per_paragraph = random.randint(3, 8)
            document = generate_dummy_document(num_paragraphs, sentences_per_paragraph)
            
            # Save document
            filename = os.path.join(split_dir, f"doc_{i:04d}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(document)
        
        logger.info(f"Created {num_docs} {split} documents in {split_dir}")
    
    logger.info(f"Dummy data generation complete in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy training data for EdgeFormer")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the dummy data")
    parser.add_argument("--num_train_docs", type=int, default=100,
                        help="Number of training documents to generate")
    parser.add_argument("--num_valid_docs", type=int, default=20,
                        help="Number of validation documents to generate")
    parser.add_argument("--num_test_docs", type=int, default=20,
                        help="Number of test documents to generate")
    
    args = parser.parse_args()
    main(args)