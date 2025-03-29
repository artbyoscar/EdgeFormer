import argparse
import torch
import logging
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer, EdgeFormerConfig
from src.utils.improved_value_estimator import ImprovedValueEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('edgeformer')
logger.info('Starting EdgeFormer test...')

def test_value_integration(args):
    """Test integration of Value Estimator with EdgeFormer"""
    
    # Create a small model for testing
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=args.hidden_size,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 2,
        max_position_embeddings=512,
        attention_type=args.model_type
    )
    
    model = EdgeFormer(config)
    model.to(args.device)
    
    # Create a value estimator
    value_estimator = ImprovedValueEstimator(config.hidden_size, config)
    value_estimator.to(args.device)
    
    # Generate random input
    input_ids = torch.randint(0, 1000, (1, args.sequence_length), device=args.device)
    
    # Forward pass with hidden states
    logger.info(f"Running forward pass on sequence of length {args.sequence_length}...")
    logits, hidden_states = model.forward_with_hidden_states(input_ids)
    
    # Test value estimation
    logger.info("Testing value estimation...")
    for i, hidden_state in enumerate(hidden_states):
        value = value_estimator(hidden_state).mean().item()
        logger.info(f"Layer {i} hidden state value: {value:.4f}")
    
    # Test convergence detection
    logger.info("Testing convergence detection with recurrent processing...")
    
    # Reset value estimator
    value_estimator.reset()
    
    # Get last hidden state
    last_hidden = hidden_states[-1][:, -1:, :]
    current_hidden = last_hidden.clone()
    
    # Simulate recurrent processing
    for i in range(args.max_iterations):
        # Estimate value
        value = value_estimator(current_hidden).mean().item()
        
        # Check convergence
        should_continue = value_estimator.should_continue_iteration(
            current_hidden, 
            i, 
            args.min_iterations, 
            args.max_iterations
        )
        
        logger.info(f"Iteration {i+1}: value = {value:.4f}, continue = {should_continue}")
        
        if not should_continue:
            logger.info(f"Converged after {i+1} iterations")
            break
            
        # Apply recurrent processing using the last transformer layer
        current_hidden = model.layers[-1].forward(current_hidden)[0]
    
    logger.info("Value integration test completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Test Value Integration with EdgeFormer')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden state size')
    parser.add_argument('--model_type', type=str, default='mla',
                        choices=['standard', 'mla', 'mla_window'],
                        help='Attention type')
    parser.add_argument('--sequence_length', type=int, default=128,
                        help='Input sequence length')
    parser.add_argument('--min_iterations', type=int, default=2,
                        help='Minimum iterations for convergence test')
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='Maximum iterations for convergence test')
    parser.add_argument('--convergence_threshold', type=float, default=0.005,
                        help='Convergence threshold')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda)')
    
    args = parser.parse_args()
    test_value_integration(args)

if __name__ == "__main__":
    main()