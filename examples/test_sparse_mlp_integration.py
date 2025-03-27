import torch
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sparse_mlp_integration_test")

def test_sparse_mlp_in_model(batch_size, seq_length, sparsity):
    """Test SparseMLP integrated in the full EdgeFormer model."""
    logger.info(f"Testing SparseMLP in EdgeFormer: batch_size={batch_size}, "
                f"seq_length={seq_length}, sparsity={sparsity}")
    
    # Create a configuration with sparse MLP enabled
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=256, 
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        mlp_sparsity=sparsity
    )

    # Initialize the model
    model = EdgeFormer(config)
    
    # Test in both training and eval modes
    for mode in ['train', 'eval']:
        logger.info(f"Testing in {mode} mode")
        
        if mode == 'train':
            model.train()
        else:
            model.eval()
        
        # Create input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logger.info(f"Success! Output logits shape: {outputs['logits'].shape}")
            
            # Check output dimensions
            expected_logits_shape = (batch_size, seq_length, config.vocab_size)
            assert outputs['logits'].shape == expected_logits_shape, \
                f"Expected logits shape {expected_logits_shape}, got {outputs['logits'].shape}"
            
        except Exception as e:
            logger.error(f"Error in {mode} mode: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting SparseMLP integration tests")
    
    # Test with different sparsity levels
    for sparsity in [0.0, 0.3, 0.5]:
        test_sparse_mlp_in_model(1, 32, sparsity)
    
    # Test with different batch sizes and sequence lengths
    test_sparse_mlp_in_model(2, 64, 0.3)
    test_sparse_mlp_in_model(4, 128, 0.3)
    
    logger.info("All integration tests completed")