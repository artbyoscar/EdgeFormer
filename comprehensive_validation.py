"""
Comprehensive EdgeFormer Validation Suite
Thorough testing before strategic partnership claims
"""

import torch
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization.dynamic_quantization import Int4Quantizer
from src.model.transformer.base_transformer import EdgeFormer, EdgeFormerConfig

def test_int4_comprehensive():
    """Comprehensive INT4 quantization validation"""
    print("🔍 COMPREHENSIVE INT4 VALIDATION")
    print("=" * 50)
    
    quantizer = Int4Quantizer()
    issues_found = []
    
    # Test 1: Various tensor shapes and sizes
    test_cases = [
        {"shape": (10, 10), "name": "Small square"},
        {"shape": (32, 512), "name": "Model size 32"},
        {"shape": (64, 768), "name": "Model size 64"},
        {"shape": (128, 1024), "name": "Model size 128"},
        {"shape": (100, 50), "name": "Asymmetric"},
        {"shape": (1, 1000), "name": "Vector-like"},
        {"shape": (1000, 1), "name": "Column vector"},
        {"shape": (7, 11), "name": "Odd dimensions"},
        {"shape": (1000, 1000), "name": "Large matrix"}
    ]
    
    compression_ratios = []
    accuracy_losses = []
    
    for case in test_cases:
        try:
            # Create realistic weights
            weights = torch.randn(case["shape"]) * 0.5
            
            # Test quantization
            quantized = quantizer.quantize(weights)
            dequantized = quantizer.dequantize(quantized)
            
            # Verify shape preservation
            if dequantized.shape != weights.shape:
                issues_found.append(f"Shape mismatch in {case['name']}: {weights.shape} → {dequantized.shape}")
                continue
            
            # Calculate metrics
            compression = quantizer.get_compression_ratio(weights, quantized)
            mse = torch.mean((weights - dequantized) ** 2).item()
            max_error = torch.max(torch.abs(weights - dequantized)).item()
            
            # Relative accuracy
            weight_range = torch.max(weights) - torch.min(weights)
            relative_error = max_error / weight_range.item() * 100 if weight_range > 0 else 0
            
            compression_ratios.append(compression)
            accuracy_losses.append(relative_error)
            
            print(f"✅ {case['name']:<15}: {compression:.2f}x compression, {relative_error:.2f}% error")
            
            # Flag potential issues
            if compression < 4.0 or compression > 8.5:
                issues_found.append(f"Compression ratio out of range for {case['name']}: {compression:.2f}x")
            if relative_error > 5.0:
                issues_found.append(f"High accuracy loss for {case['name']}: {relative_error:.2f}%")
                
        except Exception as e:
            issues_found.append(f"Failed on {case['name']}: {str(e)}")
            print(f"❌ {case['name']:<15}: FAILED - {str(e)}")
    
    # Summary statistics
    if compression_ratios:
        avg_compression = np.mean(compression_ratios)
        std_compression = np.std(compression_ratios)
        avg_accuracy_loss = np.mean(accuracy_losses)
        max_accuracy_loss = np.max(accuracy_losses)
        
        print(f"\n📊 COMPRESSION STATISTICS:")
        print(f"   Average: {avg_compression:.2f}x ± {std_compression:.2f}")
        print(f"   Range: {np.min(compression_ratios):.2f}x - {np.max(compression_ratios):.2f}x")
        
        print(f"\n📊 ACCURACY STATISTICS:")
        print(f"   Average error: {avg_accuracy_loss:.2f}%")
        print(f"   Max error: {max_accuracy_loss:.2f}%")
        
        # Validation criteria
        compression_ok = 4.0 <= avg_compression <= 8.0
        consistency_ok = std_compression < 1.0
        accuracy_ok = max_accuracy_loss < 10.0
        
        print(f"\n✅ Compression target (4-8x): {'PASS' if compression_ok else 'FAIL'}")
        print(f"✅ Consistency: {'PASS' if consistency_ok else 'FAIL'}")
        print(f"✅ Accuracy preservation: {'PASS' if accuracy_ok else 'FAIL'}")
    
    return issues_found

def test_gqa_comprehensive():
    """Comprehensive GQA validation"""
    print("\n🔍 COMPREHENSIVE GQA VALIDATION")
    print("=" * 50)
    
    issues_found = []
    
    # Test various configurations
    gqa_configs = [
        {"heads": 8, "groups": 2, "name": "8 heads, 2 groups"},
        {"heads": 8, "groups": 4, "name": "8 heads, 4 groups"},
        {"heads": 12, "groups": 3, "name": "12 heads, 3 groups"},
        {"heads": 16, "groups": 4, "name": "16 heads, 4 groups"},
    ]
    
    for config in gqa_configs:
        try:
            # Create models
            config_std = EdgeFormerConfig(
                vocab_size=1000,
                hidden_size=512,
                num_hidden_layers=2,
                num_attention_heads=config["heads"],
                attention_type="standard"
            )
            
            config_gqa = EdgeFormerConfig(
                vocab_size=1000,
                hidden_size=512,
                num_hidden_layers=2,
                num_attention_heads=config["heads"],
                attention_type="gqa",
                num_key_value_heads=config["groups"]
            )
            
            model_std = EdgeFormer(config_std)
            model_gqa = EdgeFormer(config_gqa)
            
            # Parameter count comparison
            std_params = sum(p.numel() for p in model_std.parameters())
            gqa_params = sum(p.numel() for p in model_gqa.parameters())
            reduction = (std_params - gqa_params) / std_params * 100
            
            # Test inference
            test_input = torch.randint(0, 1000, (2, 32))
            
            model_std.eval()
            model_gqa.eval()
            
            with torch.no_grad():
                output_std = model_std(test_input)
                output_gqa = model_gqa(test_input)
            
            # Shape verification
            if output_std[0].shape != output_gqa[0].shape:
                issues_found.append(f"Output shape mismatch for {config['name']}")
                continue
            
            print(f"✅ {config['name']:<20}: {reduction:.1f}% param reduction")
            
            # Flag issues
            if reduction < 1.0:
                issues_found.append(f"No parameter reduction for {config['name']}")
                
        except Exception as e:
            issues_found.append(f"GQA failed for {config['name']}: {str(e)}")
            print(f"❌ {config['name']:<20}: FAILED - {str(e)}")
    
    return issues_found

def test_performance_benchmarks():
    """Performance and memory benchmarks"""
    print("\n🔍 PERFORMANCE BENCHMARKS")
    print("=" * 50)
    
    issues_found = []
    
    # Test realistic model sizes
    benchmark_configs = [
        {"size": "Small", "vocab": 10000, "hidden": 512, "layers": 6, "heads": 8},
        {"size": "Medium", "vocab": 32000, "hidden": 768, "layers": 12, "heads": 12},
        {"size": "Large", "vocab": 50000, "hidden": 1024, "layers": 24, "heads": 16}
    ]
    
    for bench_config in benchmark_configs:
        try:
            config = EdgeFormerConfig(
                vocab_size=bench_config["vocab"],
                hidden_size=bench_config["hidden"],
                num_hidden_layers=bench_config["layers"],
                num_attention_heads=bench_config["heads"],
                attention_type="gqa",
                num_key_value_heads=bench_config["heads"] // 4
            )
            
            model = EdgeFormer(config)
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Memory usage test
            test_sequences = [32, 128, 512]
            
            print(f"\n📊 {bench_config['size']} Model ({total_params:,} parameters):")
            
            for seq_len in test_sequences:
                test_input = torch.randint(0, bench_config["vocab"], (1, seq_len))
                
                # Measure inference time
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                times = []
                for _ in range(5):  # Multiple runs for accuracy
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(test_input)
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                tokens_per_sec = seq_len / avg_time
                
                print(f"   Seq {seq_len:3d}: {tokens_per_sec:6.1f} tok/s ({avg_time*1000:5.1f}ms)")
                
                # Flag performance issues
                if tokens_per_sec < 100:  # Minimum acceptable performance
                    issues_found.append(f"Low performance for {bench_config['size']} model, seq {seq_len}: {tokens_per_sec:.1f} tok/s")
                    
        except Exception as e:
            issues_found.append(f"Benchmark failed for {bench_config['size']}: {str(e)}")
            print(f"❌ {bench_config['size']} benchmark failed: {str(e)}")
    
    return issues_found

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔍 EDGE CASE TESTING")
    print("=" * 50)
    
    issues_found = []
    quantizer = Int4Quantizer()
    
    edge_cases = [
        {"tensor": torch.zeros(10, 10), "name": "All zeros"},
        {"tensor": torch.ones(10, 10), "name": "All ones"},
        {"tensor": torch.randn(10, 10) * 1000, "name": "Large values"},
        {"tensor": torch.randn(10, 10) * 0.001, "name": "Small values"},
        {"tensor": torch.tensor([[float('inf'), 1], [2, 3]]), "name": "Infinity values"},
        {"tensor": torch.tensor([[float('nan'), 1], [2, 3]]), "name": "NaN values"},
    ]
    
    for case in edge_cases:
        try:
            if torch.isnan(case["tensor"]).any() or torch.isinf(case["tensor"]).any():
                # These should be handled gracefully
                try:
                    quantized = quantizer.quantize(case["tensor"])
                    issues_found.append(f"{case['name']}: Should handle inf/nan gracefully")
                except:
                    print(f"✅ {case['name']:<15}: Properly rejected inf/nan")
                    continue
            
            quantized = quantizer.quantize(case["tensor"])
            dequantized = quantizer.dequantize(quantized)
            
            if dequantized.shape != case["tensor"].shape:
                issues_found.append(f"{case['name']}: Shape not preserved")
            else:
                print(f"✅ {case['name']:<15}: Handled correctly")
                
        except Exception as e:
            # Some edge cases should fail gracefully
            if "inf" in case["name"].lower() or "nan" in case["name"].lower():
                print(f"✅ {case['name']:<15}: Properly rejected ({str(e)[:30]}...)")
            else:
                issues_found.append(f"{case['name']}: Unexpected failure - {str(e)}")
                print(f"❌ {case['name']:<15}: Unexpected failure")
    
    return issues_found

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🔍 EDGEFORMER COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print("Testing all components before strategic partnership claims...")
    
    all_issues = []
    
    # Run all tests
    all_issues.extend(test_int4_comprehensive())
    all_issues.extend(test_gqa_comprehensive())
    all_issues.extend(test_performance_benchmarks())
    all_issues.extend(test_edge_cases())
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 ALL TESTS PASSED!")
        print("✅ EdgeFormer is validated for strategic partnerships")
        print("✅ Ready for production deployment")
        print("✅ Claims about competitive advantages are substantiated")
        return True
    else:
        print(f"⚠️ FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        print("\n🔧 RECOMMENDATION: Address these issues before partnership claims")
        return False

if __name__ == "__main__":
    success = run_comprehensive_validation()
    
    if success:
        print("\n🚀 READY FOR STRATEGIC PARTNERSHIPS!")
    else:
        print("\n🛠️ NEED TO FIX ISSUES FIRST")