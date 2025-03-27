# examples/quantization_demo.py
import torch
import sys
import os
import time

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def apply_dynamic_quantization(model):
    """Apply PyTorch's dynamic quantization to the model."""
    try:
        # Check if quantization is supported
        if not hasattr(torch.quantization, 'quantize_dynamic'):
            print("PyTorch dynamic quantization not available.")
            return model
            
        # Quantize the model
        print("Applying INT8 dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        print("Quantization successful")
        return quantized_model
    except Exception as e:
        print(f"Quantization failed: {e}")
        return model

def benchmark_model(model, input_ids, attention_mask, name="Model", num_runs=5):
    """Benchmark a model's inference time and memory usage."""
    # Warmup
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Measure inference time
    total_time = 0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            total_time += (end_time - start_time)
    
    avg_time = total_time / num_runs
    
    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
    
    print(f"\n{name} Performance:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Parameter Memory: {param_size:.2f} MB")
    print(f"  Average inference time: {avg_time:.4f} seconds")
    
    return avg_time, param_size

def main():
    print("\n=== EdgeFormer Quantization Demo ===")
    
    # Create a model for testing
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024
    )
    
    # Initialize model and input
    model = EdgeFormer(config)
    seq_len = 128
    input_ids = torch.randint(0, 1000, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))
    
    # Benchmark FP32 model
    fp32_time, fp32_size = benchmark_model(model, input_ids, attention_mask, name="FP32 Model")
    
    # Apply dynamic quantization
    quantized_model = apply_dynamic_quantization(model)
    
    # Benchmark quantized model
    int8_time, int8_size = benchmark_model(quantized_model, input_ids, attention_mask, name="INT8 Model")
    
    # Show comparison
    if int8_size < fp32_size:
        print("\nPerformance Comparison:")
        print(f"  Memory reduction: {fp32_size / int8_size:.2f}x")
        print(f"  Speed impact: {(int8_time - fp32_time) / fp32_time * 100:.1f}% {'slower' if int8_time > fp32_time else 'faster'}")
    
    print("\nNext steps for quantization:")
    print("1. Implement static quantization with calibration")
    print("2. Try different quantization schemes (e.g., weight-only)")
    print("3. Compare different bit widths (INT8 vs INT4)")
    print("4. Evaluate model quality after quantization")

if __name__ == "__main__":
    main()