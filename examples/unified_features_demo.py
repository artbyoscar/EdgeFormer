import argparse
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.value_estimator import ImprovedValueEstimator
from src.utils.htps_budget_manager import HTPSBudgetManager
from src.utils.kv_cache_manager import KVCacheManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('unified_demo')

def main():
    parser = argparse.ArgumentParser(description='EdgeFormer Unified Features Demo')
    parser.add_argument('--prompt', type=str, default="EdgeFormer is", help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu|cuda)')
    
    # Feature flags
    parser.add_argument('--use_kv_cache', action='store_true', help='Enable KV cache management')
    parser.add_argument('--use_recurrent', action='store_true', help='Enable value-based recurrent processing')
    parser.add_argument('--use_budget', action='store_true', help='Enable HyperTree budget forcing')
    
    # KV cache parameters
    parser.add_argument('--offload_threshold', type=int, default=1024, help='Token threshold for offloading to RAM')
    
    # Recurrent processing parameters
    parser.add_argument('--min_iterations', type=int, default=2, help='Minimum recurrent iterations')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum recurrent iterations')
    parser.add_argument('--convergence_threshold', type=float, default=0.005, help='Convergence threshold')
    
    # Budget forcing parameters
    parser.add_argument('--max_budget_tokens', type=int, default=2048, help='Maximum budget tokens')
    parser.add_argument('--extension_token', type=str, default="Wait", help='Token for extending thinking')
    parser.add_argument('--extensions', type=int, default=2, help='Maximum thinking extensions')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Initialize model
    config = EdgeFormerConfig(
        vocab_size=256,  # Character-level tokenization
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        attention_type="mla",  # Using MLA attention
        # Enable features based on arguments
        enable_budget_forcing=args.use_budget,
        max_budget_tokens=args.max_budget_tokens,
        max_thinking_extensions=args.extensions,
        extension_token=args.extension_token,
        enable_recurrent_depth=args.use_recurrent,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold
    )
    
    logger.info(f"Initializing EdgeFormer with features: KV Cache: {args.use_kv_cache}, "
                f"Recurrent Processing: {args.use_recurrent}, Budget Forcing: {args.use_budget}")
    
    model = EdgeFormer(config)
    model.to(args.device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize KV cache manager if requested
    if args.use_kv_cache:
        kv_cache_manager = KVCacheManager(
            max_batch_size=1,
            max_seq_length=1024,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=args.device,
            enable_offload=True,
            offload_threshold=args.offload_threshold
        )
        model.kv_cache_manager = kv_cache_manager
        logger.info(f"KV Cache Manager initialized with offload threshold: {args.offload_threshold}")
    
    # Initialize value estimator if using recurrent processing
    if args.use_recurrent:
        value_estimator = ImprovedValueEstimator(config.hidden_size, config)
        value_estimator.to(args.device)
        logger.info(f"Value Estimator initialized")
    
    # Initialize budget manager if using budget forcing
    if args.use_budget:
        budget_manager = HTPSBudgetManager(
            max_budget_tokens=args.max_budget_tokens,
            max_thinking_extensions=args.extensions,
            extension_token=args.extension_token,
            confidence_threshold=0.9,
            complexity_threshold=0.6
        )
        logger.info(f"Budget Manager initialized with max budget: {args.max_budget_tokens}, "
                    f"extensions: {args.extensions}")
    
    # Tokenize input
    logger.info(f"Prompt: \"{args.prompt}\"")
    tokens = [ord(c) % config.vocab_size for c in args.prompt]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
    
    # Run generation with all requested features
    if args.use_recurrent:
        # Implement recurrent generation
        pass
    elif args.use_budget:
        # Implement budget-forced generation
        pass
    else:
        # Regular generation
        generated_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        
        # Convert IDs to text
        generated_text = ""
        for token_id in generated_ids[0]:
            generated_text += chr(token_id.item() % 128)
            
        logger.info(f"Generated text: \"{generated_text}\"")
    
    # Implement visualization if requested
    if args.visualize:
        # Create visualizations based on enabled features
        pass

if __name__ == "__main__":
    main()