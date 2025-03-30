# benchmarks/simple_quantization_benchmark.py
import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import EdgeFormerModel
from src.optimization.dynamic_quantization import DynamicQuantizer

def simple_benchmark():
    """Run a simple benchmark of INT4 quantization."""
    print("Creating model...")
    config = EdgeFormerConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256
    )
    
    model = EdgeFormerModel(config)
    model.eval()
    
    # Measure original model size
    fp32_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    print(f"Original model size: {fp32_size:.2f} MB")
    
    # Create input
    input_ids = torch.randint(0, 1000, (1, 32))
    
    # Benchmark FP32
    print("Benchmarking FP32 model...")
    with torch.no_grad():
        # Warmup
        _ = model(input_ids)
        
        # Benchmark
        start = time.time()
        for _ in range(5):
            _ = model(input_ids)
        fp32_time = (time.time() - start) / 5
    
    print(f"FP32 inference time: {fp32_time*1000:.2f} ms")
    
    # INT4 quantization
    print("Applying INT4 quantization...")
    try:
        int4_model = DynamicQuantizer.quantize_model_int4(model)
        
        # Benchmark INT4
        print("Benchmarking INT4 model...")
        with torch.no_grad():
            # Warmup
            _ = int4_model(input_ids)
            
            # Benchmark
            start = time.time()
            for _ in range(5):
                _ = int4_model(input_ids)
            int4_time = (time.time() - start) / 5
        
        print(f"INT4 inference time: {int4_time*1000:.2f} ms")
        print(f"Speed ratio: {fp32_time/int4_time:.2f}x")
        
        # Get quantizer to measure size
        int4_quantizer = None
        for obj in gc.get_objects():
            if isinstance(obj, Int4Quantizer) and hasattr(obj, 'model') and obj.model is model:
                int4_quantizer = obj
                break
        
        if int4_quantizer:
            savings = int4_quantizer.get_memory_savings()
            print(f"INT4 model size: {savings['quantized_size_mb']:.2f} MB")
            print(f"Compression ratio: {savings['compression_ratio']:.2f}x")
            print(f"Size reduction: {savings['size_reduction_percent']:.2f}%")
    
    except Exception as e:
        print(f"Error during INT4 quantization: {str(e)}")

if __name__ == "__main__":
    import gc
    from src.optimization.dynamic_quantization import Int4Quantizer
    
    simple_benchmark()