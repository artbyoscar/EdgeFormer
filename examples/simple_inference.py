import os
import sys
import torch
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
from src.utils.logging import setup_logging
from src.utils.device import get_device

def main():
    # Set up logging
    logger = setup_logging(debug_mode=True)
    logger.info("Starting EdgeFormer test...")
    
    # Create a small model configuration for testing
    config = EdgeFormerConfig(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=256,   # Smaller size for testing
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        latent_size_factor=8,  # MLA parameter
        use_sliding_window=True,
        sliding_window_size=256,
        use_flash_attention=False,  # Disable for testing
        use_sparse_mlp=True,
        mlp_sparsity=0.8,
        quantization=None,  # No quantization for testing
        debug_mode=True,    # Enable debug logging
    )

    logger.info("Creating model...")
    
    print("Creating model...")
    model = EdgeFormer(config)
    
    # Use GPT-2 tokenizer for testing
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Move model to available device
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Test inference
    print("Running inference...")
    prompt = "Hello, my name is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    print("Generating text...")
    output_ids = model.generate(
        input_ids,
        max_length=20,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Memory usage test
    print("\nMemory usage analysis:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    if torch.cuda.is_available():
        print(f"GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # Compare KV cache size between traditional and MLA implementations
    print("\nKV Cache Size Analysis:")
    seq_length = 1024
    tokens_in_context = seq_length
    
    # Traditional KV cache calculation
    traditional_kv_size = 2 * tokens_in_context * config.hidden_size * config.num_attention_heads * 2 / (1024**2)  # MB
    
    # MLA KV cache calculation
    mla_kv_size = 2 * tokens_in_context * config.latent_size * 2 / (1024**2)  # MB
    
    print(f"Traditional KV cache size for {tokens_in_context} tokens: {traditional_kv_size:.2f} MB")
    print(f"MLA KV cache size for {tokens_in_context} tokens: {mla_kv_size:.2f} MB")
    print(f"Reduction factor: {traditional_kv_size/mla_kv_size:.2f}x")
    
    # Estimated max context with 2GB memory limit
    available_memory_mb = 2 * 1024  # 2GB in MB
    traditional_max_tokens = int(available_memory_mb / (traditional_kv_size / tokens_in_context))
    mla_max_tokens = int(available_memory_mb / (mla_kv_size / tokens_in_context))
    
    print(f"\nEstimated max context with 2GB memory:")
    print(f"Traditional attention: ~{traditional_max_tokens:,} tokens")
    print(f"MLA attention: ~{mla_max_tokens:,} tokens")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()