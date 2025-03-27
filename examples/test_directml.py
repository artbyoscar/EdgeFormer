# examples/test_directml.py
import torch
import numpy as np
import time
import onnxruntime as ort
import logging
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_directml")

def create_edgeformer_onnx_model():
    """Export EdgeFormer to ONNX format"""
    # Create a small EdgeFormer model
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        debug_mode=False,  # Disable debug logging for export
    )
    
    model = EdgeFormer(config)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_length = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Export model to ONNX
    logger.info("Exporting EdgeFormer to ONNX format...")
    
    torch.onnx.export(
        model, 
        (input_ids, attention_mask),
        "edgeformer.onnx", 
        input_names=["input_ids", "attention_mask"], 
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"}, 
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    logger.info("Created ONNX model: edgeformer.onnx")
    return config

def test_directml():
    # Check ONNX Runtime providers
    providers = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {providers}")
    
    if 'DmlExecutionProvider' in providers:
        logger.info("DirectML is available!")
        
        try:
            # Try to export EdgeFormer to ONNX
            config = create_edgeformer_onnx_model()
            
            # Create input data
            batch_size = 1
            seq_length = 128
            input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_length)).astype(np.int64)
            attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
            
            # Create session with CPU provider
            logger.info("Testing CPU inference...")
            cpu_session = ort.InferenceSession("edgeformer.onnx", providers=['CPUExecutionProvider'])
            
            # Warmup and time CPU inference
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            _ = cpu_session.run(None, inputs)  # Warmup
            
            start_time = time.time()
            for _ in range(5):
                cpu_session.run(None, inputs)
            cpu_time = (time.time() - start_time) / 5
            logger.info(f"CPU inference time: {cpu_time:.4f} seconds")
            
            # Create session with DirectML provider
            logger.info("Testing DirectML inference...")
            dml_session = ort.InferenceSession("edgeformer.onnx", providers=['DmlExecutionProvider'])
            
            # Warmup and time DirectML inference
            _ = dml_session.run(None, inputs)  # Warmup
            
            start_time = time.time()
            for _ in range(5):
                dml_session.run(None, inputs)
            dml_time = (time.time() - start_time) / 5
            logger.info(f"DirectML inference time: {dml_time:.4f} seconds")
            
            # Calculate speedup
            speedup = cpu_time / dml_time
            logger.info(f"DirectML speedup: {speedup:.2f}x")
            
            return True
        except Exception as e:
            logger.error(f"Error running ONNX model: {e}")
            logger.info("Falling back to simple model test...")
            
            # Create a simple model
            return create_simple_onnx_model_test()
    else:
        logger.error("DirectML is not available in ONNX Runtime providers")
        return False

def create_simple_onnx_model_test():
    # Create a simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(128, 256)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(256, 128)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 128)
    
    # Export model to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        "simple_model.onnx", 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    logger.info("Created simple ONNX model: simple_model.onnx")
    
    # Create input data
    input_data = np.random.rand(1, 128).astype(np.float32)
    
    # Create session with CPU provider
    logger.info("Testing CPU inference with simple model...")
    cpu_session = ort.InferenceSession("simple_model.onnx", providers=['CPUExecutionProvider'])
    
    # Time CPU inference
    start_time = time.time()
    for _ in range(10):
        cpu_session.run(None, {"input": input_data})
    cpu_time = (time.time() - start_time) / 10
    logger.info(f"CPU inference time: {cpu_time:.4f} seconds")
    
    # Create session with DirectML provider
    logger.info("Testing DirectML inference with simple model...")
    dml_session = ort.InferenceSession("simple_model.onnx", providers=['DmlExecutionProvider'])
    
    # Time DirectML inference
    start_time = time.time()
    for _ in range(10):
        dml_session.run(None, {"input": input_data})
    dml_time = (time.time() - start_time) / 10
    logger.info(f"DirectML inference time: {dml_time:.4f} seconds")
    
    # Calculate speedup
    speedup = cpu_time / dml_time
    logger.info(f"DirectML speedup: {speedup:.2f}x")
    
    return True

if __name__ == "__main__":
    success = test_directml()
    if success:
        logger.info("DirectML test completed successfully")
    else:
        logger.error("DirectML test failed")