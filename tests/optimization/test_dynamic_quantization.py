import unittest
import torch
import sys
import os
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.optimization import DynamicQuantizer, Int4Quantizer
from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormerModel

class TestDynamicQuantization(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        config = EdgeFormerConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32
        )
        self.model = EdgeFormerModel(config)
    
    def test_int4_packing(self):
        """Test INT4 packing and unpacking."""
        # Create a tensor with known values
        tensor = torch.tensor([-8, -4, 0, 3, 7, 1], dtype=torch.int8)
        
        # Create an Int4Quantizer instance
        quantizer = Int4Quantizer(self.model)
        
        # Test packing
        packed = quantizer._pack_int4(tensor.numpy())
        
        # Test unpacking
        unpacked = quantizer._unpack_int4(packed, tensor.shape)
        
        # Convert back to tensor for comparison
        unpacked_tensor = torch.tensor(unpacked)
        
        # Check that unpacked values match original
        self.assertTrue(torch.equal(tensor, unpacked_tensor))
    
    def test_int4_quantization(self):
        """Test INT4 quantization of a model."""
        # Get original model parameters
        original_params = sum(p.numel() for p in self.model.parameters())
        
        # Quantize model
        quantized_model = DynamicQuantizer.quantize_model_int4(self.model)
        
        # Ensure model was modified
        self.assertIsNotNone(quantized_model)
        
        # Create input
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Test inference
        with torch.no_grad():
            # Run original model
            self.model.eval()
            original_output = self.model(input_ids)
            
            # Run quantized model
            quantized_model.eval()
            quantized_output = quantized_model(input_ids)
        
        # Check outputs have same structure
        self.assertEqual(len(original_output), len(quantized_output))
        
        # Check outputs are similar but not identical (due to quantization)
        for key in original_output:
            if torch.is_tensor(original_output[key]) and torch.is_tensor(quantized_output[key]):
                # Calculate similarity
                original_flat = original_output[key].reshape(-1)
                quantized_flat = quantized_output[key].reshape(-1)
                
                if original_flat.numel() > 0:
                    # Use mean absolute error
                    mae = torch.abs(original_flat - quantized_flat).mean().item()
                    # Allow some error due to quantization
                    self.assertLess(mae, 0.1, "Output difference too large")
    
    def test_memory_savings(self):
        """Test memory savings from INT4 quantization."""
        # Quantize model
        quantized_model = DynamicQuantizer.quantize_model_int4(self.model)
        
        # Find the Int4Quantizer instance
        quantizer = None
        for obj in gc.get_objects():
            if isinstance(obj, Int4Quantizer) and hasattr(obj, 'model') and obj.model is self.model:
                quantizer = obj
                break
        
        # If we found the quantizer, check memory savings
        if quantizer:
            savings = quantizer.get_memory_savings()
            self.assertGreater(savings['compression_ratio'], 2.0, 
                              "Expected at least 2x compression ratio")
            self.assertGreater(savings['size_reduction_percent'], 50.0,
                              "Expected at least 50% size reduction")

if __name__ == "__main__":
    unittest.main()