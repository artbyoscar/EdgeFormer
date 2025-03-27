# examples/test_chunking.py
import torch
import argparse
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.long_sequence import process_long_document, memory_aware_forward

def main():
    parser = argparse.ArgumentParser(description="Test EdgeFormer chunking")
    parser.add_argument("--sequence_length", type=int, default=16384, help="Total sequence length to test")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Size of chunks")
    parser.add_argument("--overlap", type=int, default=512, help="Overlap between chunks")
    parser.add_argument("--memory_aware", action="store_true", help="Use memory-aware processing")
    parser.add_argument("--attention_type", type=str, default="standard", 
                      choices=["standard", "mla", "mla_window", "auto"], 
                      help="Attention mechanism to use")
    args = parser.parse_args()
    
    # Create a model with standard config
    config = EdgeFormerConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
        intermediate_size=1024,
        max_position_embeddings=8192,
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Create a mock input of the specified length
    input_ids = torch.randint(0, config.vocab_size, (1, args.sequence_length))
    
    print(f"Testing sequence of length {args.sequence_length} with chunk size {args.chunk_size} and overlap {args.overlap}")
    print(f"Attention type: {args.attention_type}, Memory-aware mode: {'Enabled' if args.memory_aware else 'Disabled'}")
    
    # Process using chunking
    start_time = time.time()
    try:
        if args.memory_aware:
            # Use memory-aware processing
            results = memory_aware_forward(model, input_ids, attention_type=args.attention_type)
        else:
            # Use standard chunking
            if hasattr(model, 'set_attention_type'):
                model.set_attention_type(args.attention_type)
            elif hasattr(model, 'attention_type'):
                model.attention_type = args.attention_type
                
            results = process_long_document(
                model, 
                input_ids, 
                chunk_size=args.chunk_size, 
                overlap=args.overlap
            )
            
        end_time = time.time()
        print(f"Successfully processed sequence in {end_time - start_time:.2f}s")
        print(f"Output shape: {results['logits'].shape}")
        
    except Exception as e:
        print(f"Error processing sequence: {e}")

if __name__ == "__main__":
    main()