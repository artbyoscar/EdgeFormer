# examples/test_int4_quantization.py
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.quantization import quantize_model

def measure_model_size(model):
    """Measure model size in MB accounting for quantized models"""
    # Check if this is an INT4 quantized model by looking for Int4Quantizer attributes
    is_int4 = False
    for module in model.modules():
        if hasattr(module, 'forward') and 'weight_int' in str(module.forward):
            is_int4 = True
            break

    if is_int4:
        # Get the quantizer from gc
        import gc
        quantizer = None
        for obj in gc.get_objects():
            if isinstance(obj, Int4Quantizer) and hasattr(obj, 'model') and obj.model is model:
                quantizer = obj
                break
        if quantizer and hasattr(quantizer, 'get_memory_savings'):
            savings = quantizer.get_memory_savings()
            return savings['quantized_size_mb']
        
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def test_model_accuracy(original_model, quantized_model, sequence_length=512, num_samples=10):
    """Test model accuracy before and after quantization"""
    original_model.eval()
    quantized_model.eval()
    
    mse_values = []
    similarity_values = []
    
    for _ in range(num_samples):
        # Create random input
        input_ids = torch.randint(0, 32000, (1, sequence_length))
        attention_mask = torch.ones(1, sequence_length)
        
        # Get original outputs
        with torch.no_grad():
            original_outputs = original_model(input_ids=input_ids, attention_mask=attention_mask)
            original_logits = original_outputs["logits"]
        
        # Get quantized outputs
        with torch.no_grad():
            quantized_outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
            quantized_logits = quantized_outputs["logits"]
        
        # Calculate MSE
        mse = ((original_logits - quantized_logits) ** 2).mean().item()
        mse_values.append(mse)
        
        # Calculate cosine similarity
        original_flat = original_logits.view(-1)
        quantized_flat = quantized_logits.view(-1)
        
        similarity = torch.nn.functional.cosine_similarity(
            original_flat.unsqueeze(0), 
            quantized_flat.unsqueeze(0)
        ).item()
        similarity_values.append(similarity)
    
    return {
        "mse": np.mean(mse_values),
        "similarity": np.mean(similarity_values) * 100  # Convert to percentage
    }

def benchmark_inference_speed(model, sequence_length=512, num_trials=10):
    """Benchmark inference speed"""
    model.eval()
    
    # Create input
    input_ids = torch.randint(0, 32000, (1, sequence_length))
    attention_mask = torch.ones(1, sequence_length)
    
    # Warm-up
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Benchmark
    timings = []
    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        end_time = time.time()
        timings.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(timings)
    tokens_per_sec = sequence_length / avg_time
    std_dev = np.std(timings) * 1000  # Convert to ms
    
    return {
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "std_dev_ms": std_dev,
        "raw_timings_ms": [t * 1000 for t in timings]  # Convert to ms
    }

def benchmark_sequence_lengths(model, model_name, sequence_lengths=[128, 256, 512, 1024, 2048]):
    """Benchmark model across different sequence lengths"""
    results = []
    
    for seq_len in sequence_lengths:
        print(f"Benchmarking {model_name} with sequence length {seq_len}...")
        try:
            speed = benchmark_inference_speed(model, sequence_length=seq_len, num_trials=5)
            results.append((seq_len, speed["tokens_per_sec"]))
            print(f"  → {speed['tokens_per_sec']:.2f} tokens/sec")
        except RuntimeError as e:
            print(f"  → Failed with error: {str(e)}")
            break
    
    return results

def plot_results(model_sizes, accuracy_results, speed_results, throughput_results, output_dir="benchmark_results"):
    """Plot benchmark results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Model size comparison
    plt.figure(figsize=(10, 6))
    models = ["FP32", "INT8", "INT4"]
    sizes = [model_sizes["fp32"], model_sizes["int8"], model_sizes["int4"]]
    compression = [1.0, model_sizes["fp32"] / model_sizes["int8"], model_sizes["fp32"] / model_sizes["int4"]]
    
    # Create subplot for model sizes
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, sizes, color=["blue", "orange", "green"])
    plt.title("Model Size Comparison")
    plt.ylabel("Size (MB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add size labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{sizes[i]:.1f} MB", ha="center")
    
    # Create subplot for compression ratios
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, compression, color=["blue", "orange", "green"])
    plt.title("Compression Ratio")
    plt.ylabel("Ratio (higher is better)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add compression labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f"{compression[i]:.1f}x", ha="center")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/size_comparison_{timestamp}.png")
    
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    models = ["INT8", "INT4"]
    mse = [accuracy_results["int8"]["mse"], accuracy_results["int4"]["mse"]]
    similarity = [accuracy_results["int8"]["similarity"], accuracy_results["int4"]["similarity"]]
    
    # Create subplot for MSE (lower is better)
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, mse, color=["orange", "green"])
    plt.title("Mean Squared Error (lower is better)")
    plt.ylabel("MSE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add MSE labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f"{mse[i]:.6f}", ha="center")
    
    # Create subplot for similarity (higher is better)
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, similarity, color=["orange", "green"])
    plt.title("Output Similarity (higher is better)")
    plt.ylabel("Similarity (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add similarity labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{similarity[i]:.1f}%", ha="center")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison_{timestamp}.png")
    
    # Inference speed comparison
    plt.figure(figsize=(10, 6))
    models = ["FP32", "INT8", "INT4"]
    tokens_per_sec = [
        speed_results["fp32"]["tokens_per_sec"], 
        speed_results["int8"]["tokens_per_sec"], 
        speed_results["int4"]["tokens_per_sec"]
    ]
    speedup = [
        1.0, 
        speed_results["int8"]["tokens_per_sec"] / speed_results["fp32"]["tokens_per_sec"], 
        speed_results["int4"]["tokens_per_sec"] / speed_results["fp32"]["tokens_per_sec"]
    ]
    
    # Create subplot for tokens per second
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, tokens_per_sec, color=["blue", "orange", "green"])
    plt.title("Inference Speed")
    plt.ylabel("Tokens per Second")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add speed labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f"{tokens_per_sec[i]:.0f}", ha="center")
    
    # Create subplot for speedup
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, speedup, color=["blue", "orange", "green"])
    plt.title("Speed Improvement")
    plt.ylabel("Speedup Factor")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add speedup labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f"{speedup[i]:.2f}x", ha="center")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speed_comparison_{timestamp}.png")
    
    # Throughput across sequence lengths
    plt.figure(figsize=(12, 6))
    
    # Plot for each model type
    for model_name, results in throughput_results.items():
        if results:  # Check if we have results
            x = [r[0] for r in results]  # Sequence lengths
            y = [r[1] for r in results]  # Tokens per second
            
            if model_name == "fp32":
                plt.plot(x, y, 'o-', label="FP32", color="blue", linewidth=2)
            elif model_name == "int8":
                plt.plot(x, y, 's-', label="INT8", color="orange", linewidth=2)
            elif model_name == "int4":
                plt.plot(x, y, '^-', label="INT4", color="green", linewidth=2)
    
    plt.title("Throughput vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens per Second")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.xscale("log", base=2)  # Log scale for sequence length
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_comparison_{timestamp}.png")
    
    # Save results to CSV
    with open(f"{output_dir}/benchmark_results_{timestamp}.csv", "w") as f:
        # Header
        f.write("Metric,FP32,INT8,INT4\n")
        
        # Model size
        f.write(f"Model Size (MB),{model_sizes['fp32']:.2f},{model_sizes['int8']:.2f},{model_sizes['int4']:.2f}\n")
        f.write(f"Compression Ratio,1.00,{model_sizes['fp32'] / model_sizes['int8']:.2f},{model_sizes['fp32'] / model_sizes['int4']:.2f}\n")
        
        # Accuracy
        f.write(f"MSE,0.0,{accuracy_results['int8']['mse']:.6f},{accuracy_results['int4']['mse']:.6f}\n")
        f.write(f"Similarity (%),100.0,{accuracy_results['int8']['similarity']:.2f},{accuracy_results['int4']['similarity']:.2f}\n")
        
        # Speed
        f.write(f"Tokens/sec,{speed_results['fp32']['tokens_per_sec']:.2f},{speed_results['int8']['tokens_per_sec']:.2f},{speed_results['int4']['tokens_per_sec']:.2f}\n")
        f.write(f"Speedup,1.00,{speed_results['int8']['tokens_per_sec'] / speed_results['fp32']['tokens_per_sec']:.2f},{speed_results['int4']['tokens_per_sec'] / speed_results['fp32']['tokens_per_sec']:.2f}\n")

def main():
    """Main function to run all benchmarks"""
    print("Starting EdgeFormer quantization benchmarks...")
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"benchmark_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model configuration - using latent_size_factor instead of latent_size
    print("\nCreating model...")
    config = EdgeFormerConfig(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        latent_size_factor=8  # This creates a latent size of 96 (768/8)
    )
    model = EdgeFormer(config)
    model.eval()
    
    # Create INT8 and INT4 quantized models
    print("\nQuantizing model to INT8...")
    int8_model = quantize_model(model, quantization_type="int8")
    print("\nQuantizing model to INT4...")
    int4_model = quantize_model(model, quantization_type="int4")
    
    # Measure model sizes
    print("\nMeasuring model sizes...")
    fp32_size = measure_model_size(model)
    int8_size = measure_model_size(int8_model)
    int4_size = measure_model_size(int4_model)
    
    model_sizes = {
        "fp32": fp32_size,
        "int8": int8_size,
        "int4": int4_size
    }
    
    print(f"FP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {int8_size:.2f} MB (reduction: {fp32_size / int8_size:.2f}x)")
    print(f"INT4 model size: {int4_size:.2f} MB (reduction: {fp32_size / int4_size:.2f}x)")
    
    # Test model accuracy
    print("\nTesting model accuracy...")
    int8_accuracy = test_model_accuracy(model, int8_model)
    int4_accuracy = test_model_accuracy(model, int4_model)
    
    accuracy_results = {
        "int8": int8_accuracy,
        "int4": int4_accuracy
    }
    
    print(f"INT8 accuracy - MSE: {int8_accuracy['mse']:.6f}, similarity: {int8_accuracy['similarity']:.2f}%")
    print(f"INT4 accuracy - MSE: {int4_accuracy['mse']:.6f}, similarity: {int4_accuracy['similarity']:.2f}%")
    
    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    fp32_speed = benchmark_inference_speed(model)
    int8_speed = benchmark_inference_speed(int8_model)
    int4_speed = benchmark_inference_speed(int4_model)
    
    speed_results = {
        "fp32": fp32_speed,
        "int8": int8_speed,
        "int4": int4_speed
    }
    
    print(f"FP32 speed: {fp32_speed['tokens_per_sec']:.2f} tokens/sec")
    print(f"INT8 speed: {int8_speed['tokens_per_sec']:.2f} tokens/sec (speedup: {int8_speed['tokens_per_sec'] / fp32_speed['tokens_per_sec']:.2f}x)")
    print(f"INT4 speed: {int4_speed['tokens_per_sec']:.2f} tokens/sec (speedup: {int4_speed['tokens_per_sec'] / fp32_speed['tokens_per_sec']:.2f}x)")
    
    # Benchmark across different sequence lengths
    print("\nBenchmarking different sequence lengths...")
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    print("\nFP32 model:")
    fp32_throughput = benchmark_sequence_lengths(model, "FP32", sequence_lengths)
    
    print("\nINT8 model:")
    int8_throughput = benchmark_sequence_lengths(int8_model, "INT8", sequence_lengths)
    
    print("\nINT4 model:")
    int4_throughput = benchmark_sequence_lengths(int4_model, "INT4", sequence_lengths)
    
    throughput_results = {
        "fp32": fp32_throughput,
        "int8": int8_throughput,
        "int4": int4_throughput
    }
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(model_sizes, accuracy_results, speed_results, throughput_results, output_dir)
    
    print(f"\nBenchmark complete! Results saved to {output_dir}/")
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"FP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {int8_size:.2f} MB (reduction: {fp32_size / int8_size:.2f}x)")
    print(f"INT4 model size: {int4_size:.2f} MB (reduction: {fp32_size / int4_size:.2f}x)")
    print(f"INT8 accuracy: {int8_accuracy['similarity']:.2f}% similar to FP32")
    print(f"INT4 accuracy: {int4_accuracy['similarity']:.2f}% similar to FP32")
    print(f"FP32 speed: {fp32_speed['tokens_per_sec']:.2f} tokens/sec")
    print(f"INT8 speed: {int8_speed['tokens_per_sec']:.2f} tokens/sec (speedup: {int8_speed['tokens_per_sec'] / fp32_speed['tokens_per_sec']:.2f}x)")
    print(f"INT4 speed: {int4_speed['tokens_per_sec']:.2f} tokens/sec (speedup: {int4_speed['tokens_per_sec'] / fp32_speed['tokens_per_sec']:.2f}x)")
    
    max_fp32_seq_len = fp32_throughput[-1][0] if fp32_throughput else "N/A"
    max_int8_seq_len = int8_throughput[-1][0] if int8_throughput else "N/A"
    max_int4_seq_len = int4_throughput[-1][0] if int4_throughput else "N/A"
    
    print(f"Max sequence length (FP32): {max_fp32_seq_len}")
    print(f"Max sequence length (INT8): {max_int8_seq_len}")
    print(f"Max sequence length (INT4): {max_int4_seq_len}")

if __name__ == "__main__":
    main()