import torch
from src.model.config import EdgeFormerConfig
from src.model.sparse_mlp import SparseMLP
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sparse_mlp_test")

def test_sparse_mlp_dimensions(batch_size, seq_length, hidden_size, intermediate_size, sparsity):
    """Test SparseMLP with specific dimensions and sparsity."""
    logger.info(f"Testing SparseMLP with: batch_size={batch_size}, seq_length={seq_length}, " 
                f"hidden_size={hidden_size}, intermediate_size={intermediate_size}, sparsity={sparsity}")
    
    # Create config
    config = EdgeFormerConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=0.1,
        mlp_sparsity=sparsity
    )
    
    # Create a SparseMLP instance
    sparse_mlp = SparseMLP(config)
    
    # Create random input tensor
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    # Try forward pass
    try:
        # Run both in training and eval modes
        sparse_mlp.train()
        train_output = sparse_mlp(hidden_states)
        logger.info(f"Training mode success! Output shape: {train_output.shape}")
        
        sparse_mlp.eval()
        eval_output = sparse_mlp(hidden_states)
        logger.info(f"Eval mode success! Output shape: {eval_output.shape}")
        
        # Check output dimensions
        expected_shape = (batch_size, seq_length, hidden_size)
        assert train_output.shape == expected_shape, f"Expected {expected_shape}, got {train_output.shape}"
        assert eval_output.shape == expected_shape, f"Expected {expected_shape}, got {eval_output.shape}"
        
        # Check sparsity percentage in intermediate activations
        sparse_mlp.train()
        with torch.no_grad():
            intermediate = sparse_mlp.dense_h_to_4h(hidden_states)
            
            # Apply activation sparsity if needed
            if hasattr(sparse_mlp, 'get_activation_mask') and sparsity > 0:
                act_mask = sparse_mlp.get_activation_mask(intermediate.shape, intermediate.device)
                intermediate = intermediate * act_mask
                
            zeros_percentage = (intermediate == 0).float().mean().item() * 100
            logger.info(f"Zeros percentage in intermediate activations: {zeros_percentage:.2f}%")
            
        return True
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting SparseMLP tests")
    
    # Test with fixed dimensions and different sparsity levels
    sparsity_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
    for sparsity in sparsity_levels:
        test_sparse_mlp_dimensions(2, 16, 128, 512, sparsity)
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        test_sparse_mlp_dimensions(batch_size, 32, 128, 512, 0.3)
    
    # Test with different sequence lengths
    for seq_length in [8, 16, 32, 64, 128]:
        test_sparse_mlp_dimensions(2, seq_length, 128, 512, 0.3)
    
    # Test with different hidden and intermediate sizes
    test_sparse_mlp_dimensions(2, 32, 256, 1024, 0.3)
    test_sparse_mlp_dimensions(2, 32, 384, 1536, 0.3)
    test_sparse_mlp_dimensions(2, 32, 512, 2048, 0.3)
    
    logger.info("All tests completed")