#!/usr/bin/env python3
"""
Compression utilities for EdgeFormer
Supporting functions for model compression and analysis
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

def calculate_model_size(model: nn.Module) -> float:
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in megabytes
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    # Assume float32 (4 bytes per parameter)
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def analyze_layer_sizes(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Analyze individual layer sizes and types
    
    Args:
        model: PyTorch model
        
    Returns:
        List of layer information dictionaries
    """
    layer_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            param_size_mb = (param_count * 4) / (1024 * 1024)  # Assume float32
            
            layer_info.append({
                'name': name,
                'type': type(module).__name__,
                'parameters': param_count,
                'size_mb': param_size_mb,
                'module': module
            })
    
    # Sort by size (largest first)
    layer_info.sort(key=lambda x: x['parameters'], reverse=True)
    return layer_info

def get_compression_candidates(model: nn.Module, min_size_mb: float = 0.1) -> List[Dict[str, Any]]:
    """
    Identify layers that are good candidates for compression
    
    Args:
        model: PyTorch model
        min_size_mb: Minimum layer size to consider for compression
        
    Returns:
        List of compression candidate layers
    """
    candidates = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Embedding)):
            param_count = sum(p.numel() for p in module.parameters())
            param_size_mb = (param_count * 4) / (1024 * 1024)
            
            if param_size_mb >= min_size_mb:
                candidates.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': param_count,
                    'size_mb': param_size_mb,
                    'module': module,
                    'compressible': True
                })
    
    candidates.sort(key=lambda x: x['size_mb'], reverse=True)
    return candidates

def estimate_compression_benefit(model: nn.Module, compression_ratio: float = 8.0) -> Dict[str, Any]:
    """
    Estimate the benefit of compressing a model
    
    Args:
        model: PyTorch model
        compression_ratio: Expected compression ratio
        
    Returns:
        Dictionary with compression estimates
    """
    original_size = calculate_model_size(model)
    candidates = get_compression_candidates(model)
    
    compressible_size = sum(layer['size_mb'] for layer in candidates)
    non_compressible_size = original_size - compressible_size
    
    compressed_size = (compressible_size / compression_ratio) + non_compressible_size
    total_compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    size_reduction_mb = original_size - compressed_size
    size_reduction_percent = (size_reduction_mb / original_size) * 100
    
    return {
        'original_size_mb': original_size,
        'compressed_size_mb': compressed_size,
        'size_reduction_mb': size_reduction_mb,
        'size_reduction_percent': size_reduction_percent,
        'total_compression_ratio': total_compression_ratio,
        'compressible_layers': len(candidates),
        'compressible_size_mb': compressible_size,
        'non_compressible_size_mb': non_compressible_size
    }

def benchmark_inference_speed(model: nn.Module, 
                            input_shape: Tuple[int, ...], 
                            num_runs: int = 100,
                            warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, ...)
        num_runs: Number of inference runs for timing
        warmup_runs: Number of warmup runs (not counted)
        
    Returns:
        Dictionary with timing results
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Actual timing runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'median_time_ms': np.median(times),
        'throughput_samples_per_sec': input_shape[0] / (np.mean(times) / 1000)
    }

def compare_model_outputs(model1: nn.Module, 
                         model2: nn.Module, 
                         input_tensor: torch.Tensor,
                         tolerance: float = 1e-3) -> Dict[str, Any]:
    """
    Compare outputs between two models (e.g., original vs compressed)
    
    Args:
        model1: First model (e.g., original)
        model2: Second model (e.g., compressed)
        input_tensor: Input to both models
        tolerance: Tolerance for considering outputs "similar"
        
    Returns:
        Dictionary with comparison results
    """
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
    
    # Handle different output formats
    if isinstance(output1, tuple):
        output1 = output1[0]
    if isinstance(output2, tuple):
        output2 = output2[0]
    
    # Calculate differences
    abs_diff = torch.abs(output1 - output2)
    rel_diff = abs_diff / (torch.abs(output1) + 1e-8)
    
    mse = torch.mean((output1 - output2) ** 2).item()
    mae = torch.mean(abs_diff).item()
    max_abs_diff = torch.max(abs_diff).item()
    max_rel_diff = torch.max(rel_diff).item()
    
    # Check similarity
    similar_within_tolerance = torch.all(abs_diff < tolerance).item()
    percent_similar = (torch.sum(abs_diff < tolerance).float() / abs_diff.numel()).item() * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'max_absolute_difference': max_abs_diff,
        'max_relative_difference': max_rel_diff,
        'similar_within_tolerance': similar_within_tolerance,
        'percent_similar': percent_similar,
        'tolerance_used': tolerance,
        'output_shape': list(output1.shape)
    }

def save_compression_report(results: Dict[str, Any], 
                          output_path: str = "compression_report.json") -> Path:
    """
    Save compression analysis results to a JSON file
    
    Args:
        results: Compression results dictionary
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any torch tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj
    
    json_compatible_results = convert_for_json(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_compatible_results, f, indent=2)
    
    return output_path

def load_compression_report(file_path: str) -> Dict[str, Any]:
    """
    Load compression results from a JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded results dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def print_compression_summary(results: Dict[str, Any]):
    """
    Print a human-readable summary of compression results
    
    Args:
        results: Compression results dictionary
    """
    print("🔍 COMPRESSION ANALYSIS SUMMARY")
    print("=" * 50)
    
    if 'model_config' in results:
        config = results['model_config']
        print(f"📋 Model: {config.get('name', 'Unknown')}")
        if 'hidden_size' in config:
            print(f"   Hidden Size: {config['hidden_size']}")
        if 'num_layers' in config:
            print(f"   Layers: {config['num_layers']}")
    
    if 'compression' in results:
        comp = results['compression']
        print(f"\n🗜️  Compression Results:")
        print(f"   Overall Compression: {comp.get('overall_compression', 0):.1f}x")
        print(f"   Success Rate: {comp.get('success_rate', 0):.1f}%")
        print(f"   Average Accuracy Loss: {comp.get('avg_accuracy_loss', 0):.3f}%")
        print(f"   Size: {comp.get('original_size_mb', 0):.1f}MB → {comp.get('compressed_size_mb', 0):.1f}MB")
    
    if 'inference' in results:
        inf = results['inference']
        print(f"\n⚡ Performance:")
        print(f"   Inference Time: {inf.get('inference_time_ms', 0):.2f}ms")
        if 'tokens_per_second' in inf:
            print(f"   Throughput: {inf['tokens_per_second']:.1f} tokens/sec")
        elif 'images_per_second' in inf:
            print(f"   Throughput: {inf['images_per_second']:.1f} images/sec")

def create_compression_config(compression_ratio: float = 8.0,
                            exclude_layers: Optional[List[str]] = None,
                            per_channel: bool = True,
                            symmetric: bool = True) -> Dict[str, Any]:
    """
    Create a compression configuration dictionary
    
    Args:
        compression_ratio: Target compression ratio
        exclude_layers: Layers to exclude from compression
        per_channel: Use per-channel quantization
        symmetric: Use symmetric quantization
        
    Returns:
        Compression configuration dictionary
    """
    if exclude_layers is None:
        exclude_layers = ['lm_head', 'embed_tokens', 'norm', 'embedding']
    
    return {
        'compression_ratio': compression_ratio,
        'exclude_layers': exclude_layers,
        'quantization': {
            'per_channel': per_channel,
            'symmetric': symmetric,
            'bits': 4
        },
        'preserve_accuracy': True,
        'validate_outputs': True
    }

# Memory tracking utilities
def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage if available"""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
        }
    else:
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}

def reset_memory_stats():
    """Reset GPU memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    print("🛠️  Compression Utils Test")
    print("=" * 40)
    
    # Test with a simple model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 1024)
            self.linear2 = nn.Linear(1024, 512)
            self.linear3 = nn.Linear(512, 256)
    
    model = TestModel()
    
    # Test size calculation
    size = calculate_model_size(model)
    print(f"📏 Model size: {size:.2f} MB")
    
    # Test parameter counting
    params = count_parameters(model)
    print(f"🔢 Parameters: {params['total_parameters']:,}")
    
    # Test compression estimation
    estimate = estimate_compression_benefit(model, compression_ratio=8.0)
    print(f"🗜️  Estimated compression: {estimate['total_compression_ratio']:.1f}x")
    print(f"💾 Size reduction: {estimate['size_reduction_percent']:.1f}%")
    
    print("\n✅ Compression utils test complete!")
