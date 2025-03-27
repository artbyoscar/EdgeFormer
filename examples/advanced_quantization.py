# examples/advanced_quantization.py
import torch
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig

def generate_calibration_data(vocab_size=1000, seq_len=128, num_samples=10):
    """Generate random data for calibration."""
    calibration_data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        attention_mask = torch.ones((1, seq_len))
        calibration_data.append((input_ids, attention_mask))
    return calibration_data

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

def quantize_dynamic(model):
    """Apply dynamic quantization to the model."""
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def quantize_static(model, calibration_data):
    """Apply static quantization with calibration to the model."""
    # Add observers to model
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with some sample data
    for input_ids, attention_mask in calibration_data:
        model_prepared(input_ids=input_ids, attention_mask=attention_mask)
    
    # Convert to a quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized

def compare_outputs(fp32_model, int8_model, input_data):
    """Compare model outputs to assess quantization quality."""
    input_ids, attention_mask = input_data
    
    with torch.no_grad():
        fp32_output = fp32_model(input_ids=input_ids, attention_mask=attention_mask)
        int8_output = int8_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Handle dict outputs
    if isinstance(fp32_output, dict) and isinstance(int8_output, dict):
        fp32_logits = fp32_output.get('logits', list(fp32_output.values())[0])
        int8_logits = int8_output.get('logits', list(int8_output.values())[0])
    else:
        fp32_logits = fp32_output
        int8_logits = int8_output
        
    # Calculate mean squared error and cosine similarity
    mse = torch.nn.functional.mse_loss(fp32_logits, int8_logits).item()
    
    # Reshape tensors for cosine similarity calculation
    fp32_flat = fp32_logits.view(-1)
    int8_flat = int8_logits.view(-1)
    cos_sim = torch.nn.functional.cosine_similarity(fp32_flat, int8_flat, dim=0).item()
    
    return {
        'mse': mse,
        'cosine_similarity': cos_sim
    }

def main():
    print("\n=== EdgeFormer Advanced Quantization Demo ===")
    
    # Create a model for testing
    config = EdgeFormerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024
    )
    
    # Initialize model and input
    model_fp32 = EdgeFormer(config)
    seq_len = 128
    input_ids = torch.randint(0, 1000, (1, seq_len))
    attention_mask = torch.ones((1, seq_len))
    test_input = (input_ids, attention_mask)
    
    # Generate calibration data
    print("Generating calibration data...")
    calibration_data = generate_calibration_data()
    
    # Benchmark FP32 model
    print("\nBenchmarking FP32 model...")
    fp32_time, fp32_size = benchmark_model(model_fp32, input_ids, attention_mask, name="FP32 Model")
    
    # Apply dynamic INT8 quantization
    print("\nApplying dynamic INT8 quantization...")
    try:
        model_int8_dynamic = quantize_dynamic(model_fp32)
        int8_dynamic_time, int8_dynamic_size = benchmark_model(
            model_int8_dynamic, input_ids, attention_mask, name="INT8 Dynamic Model"
        )
        quality_dynamic = compare_outputs(model_fp32, model_int8_dynamic, test_input)
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
        int8_dynamic_time, int8_dynamic_size = None, None
        quality_dynamic = None
    
    # Create performance comparison
    print("\nPerformance Comparison:")
    results = {
        "Model": ["FP32"],
        "Size (MB)": [fp32_size],
        "Inference Time (s)": [fp32_time],
        "MSE": [0.0],
        "Cosine Similarity": [1.0]
    }
    
    if int8_dynamic_time is not None:
        speed_impact = (int8_dynamic_time - fp32_time) / fp32_time * 100
        results["Model"].append("INT8 Dynamic")
        results["Size (MB)"].append(int8_dynamic_size)
        results["Inference Time (s)"].append(int8_dynamic_time)
        results["MSE"].append(quality_dynamic["mse"])
        results["Cosine Similarity"].append(quality_dynamic["cosine_similarity"])
        
        print(f"  Dynamic INT8 vs FP32:")
        print(f"    Memory reduction: {fp32_size / int8_dynamic_size:.2f}x")
        print(f"    Speed impact: {speed_impact:.1f}% {'slower' if speed_impact > 0 else 'faster'}")
        print(f"    Output MSE: {quality_dynamic['mse']:.6f}")
        print(f"    Output similarity: {quality_dynamic['cosine_similarity']:.6f}")

    print("\nQuantization Next Steps:")
    print("1. Implement more advanced quantization techniques")
    print("2. Experiment with INT4 quantization")
    print("3. Try weight-only quantization which can be more stable")
    print("4. Test with various sequence lengths and more complex inputs")

if __name__ == "__main__":
    main()