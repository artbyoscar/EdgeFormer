# examples/test_int4_quantization_enhanced.py
"""
Enhanced INT4 Quantization Testing with Professional Error Handling
Improved version of your existing test_int4_quantization.py
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.model.edgeformer import EdgeFormer
    from src.model.config import EdgeFormerConfig
    from src.utils.quantization import quantize_model
    EDGEFORMER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  EdgeFormer imports failed: {e}")
    EDGEFORMER_AVAILABLE = False


class ProgressBar:
    """Simple progress bar for long operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self, increment: int = 1):
        """Update progress bar."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r{self.description}: |{bar}| {percentage:.1f}% ({self.current}/{self.total})', end='')
        if self.current >= self.total:
            print()  # New line when complete


class EnhancedQuantizationTester:
    """Enhanced quantization testing with comprehensive error handling."""
    
    def __init__(self):
        """Initialize the enhanced tester."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.errors = []
        self.start_time = datetime.now()
        
        print(f"🔧 Enhanced EdgeFormer INT4 Quantization Tester")
        print(f"   Device: {self.device}")
        print(f"   EdgeFormer Available: {'✅' if EDGEFORMER_AVAILABLE else '❌'}")
        print(f"   Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def measure_model_size(self, model: torch.nn.Module) -> float:
        """Enhanced model size measurement with error handling."""
        try:
            # Check if this is an INT4 quantized model
            is_int4 = self._detect_int4_quantization(model)
            
            if is_int4:
                # Try to get quantizer memory savings
                try:
                    quantizer = self._find_quantizer(model)
                    if quantizer and hasattr(quantizer, 'get_memory_savings'):
                        savings = quantizer.get_memory_savings()
                        return savings['quantized_size_mb']
                except Exception as e:
                    print(f"   ⚠️  Quantizer memory calculation failed: {e}")
            
            # Standard size calculation
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            size_mb = (param_size + buffer_size) / (1024 ** 2)
            
            return size_mb
            
        except Exception as e:
            error_msg = f"Model size measurement failed: {e}"
            self.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return 0.0
    
    def _detect_int4_quantization(self, model: torch.nn.Module) -> bool:
        """Detect if model uses INT4 quantization."""
        try:
            for module in model.modules():
                if hasattr(module, 'forward') and 'weight_int' in str(module.forward):
                    return True
                # Check for other INT4 indicators
                if hasattr(module, 'weight_scale') or hasattr(module, 'weight_zero_point'):
                    return True
            return False
        except:
            return False
    
    def _find_quantizer(self, model: torch.nn.Module):
        """Find the quantizer object for a model."""
        try:
            import gc
            for obj in gc.get_objects():
                if hasattr(obj, '__class__') and 'Int4Quantizer' in str(obj.__class__):
                    if hasattr(obj, 'model') and obj.model is model:
                        return obj
            return None
        except:
            return None
    
    def test_model_accuracy(self, original_model: torch.nn.Module, 
                          quantized_model: torch.nn.Module,
                          sequence_length: int = 512, 
                          num_samples: int = 10) -> Dict[str, float]:
        """Enhanced accuracy testing with comprehensive error handling."""
        print(f"🧪 Testing accuracy (sequence_length={sequence_length}, samples={num_samples})")
        
        try:
            original_model.eval()
            quantized_model.eval()
            
            mse_values = []
            similarity_values = []
            
            progress = ProgressBar(num_samples, "Accuracy Testing")
            
            for i in range(num_samples):
                try:
                    # Create random input with error handling
                    input_ids = torch.randint(0, min(32000, 10000), (1, sequence_length))
                    attention_mask = torch.ones(1, sequence_length)
                    
                    # Move to device if possible
                    try:
                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                    except:
                        pass  # Keep on CPU if device transfer fails
                    
                    # Test original model
                    with torch.no_grad():
                        try:
                            original_outputs = original_model(input_ids=input_ids, attention_mask=attention_mask)
                            if isinstance(original_outputs, dict):
                                original_logits = original_outputs["logits"]
                            else:
                                original_logits = original_outputs
                        except Exception as e:
                            # Fallback to simple forward pass
                            original_logits = original_model(input_ids)
                    
                    # Test quantized model
                    with torch.no_grad():
                        try:
                            quantized_outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
                            if isinstance(quantized_outputs, dict):
                                quantized_logits = quantized_outputs["logits"]
                            else:
                                quantized_logits = quantized_outputs
                        except Exception as e:
                            # Fallback to simple forward pass
                            quantized_logits = quantized_model(input_ids)
                    
                    # Calculate metrics
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
                    
                    progress.update()
                    
                except Exception as e:
                    error_msg = f"Sample {i} failed: {e}"
                    self.errors.append(error_msg)
                    progress.update()
                    continue
            
            if not mse_values:
                raise ValueError("No valid accuracy samples obtained")
            
            results = {
                "mse": np.mean(mse_values),
                "mse_std": np.std(mse_values),
                "similarity": np.mean(similarity_values) * 100,
                "similarity_std": np.std(similarity_values) * 100,
                "valid_samples": len(mse_values),
                "total_samples": num_samples
            }
            
            print(f"   ✅ Accuracy test completed: {len(mse_values)}/{num_samples} valid samples")
            return results
            
        except Exception as e:
            error_msg = f"Accuracy testing failed: {e}"
            self.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return {
                "mse": float('inf'),
                "similarity": 0.0,
                "valid_samples": 0,
                "error": str(e)
            }
    
    def benchmark_inference_speed(self, model: torch.nn.Module, 
                                sequence_length: int = 512, 
                                num_trials: int = 10) -> Dict[str, float]:
        """Enhanced inference speed benchmarking."""
        print(f"⏱️  Benchmarking inference speed (sequence_length={sequence_length}, trials={num_trials})")
        
        try:
            model.eval()
            
            # Create input with error handling
            input_ids = torch.randint(0, min(32000, 10000), (1, sequence_length))
            attention_mask = torch.ones(1, sequence_length)
            
            # Move to device if possible
            try:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
            except:
                pass
            
            # Warm-up runs
            print("   🔥 Performing warm-up runs...")
            for _ in range(3):
                try:
                    with torch.no_grad():
                        _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except:
                    # Fallback
                    with torch.no_grad():
                        _ = model(input_ids)
            
            # Actual benchmarking
            times = []
            progress = ProgressBar(num_trials, "Speed Benchmark")
            
            for i in range(num_trials):
                try:
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        try:
                            _ = model(input_ids=input_ids, attention_mask=attention_mask)
                        except:
                            _ = model(input_ids)
                    
                    # Synchronize if using CUDA
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    progress.update()
                    
                except Exception as e:
                    self.errors.append(f"Speed trial {i} failed: {e}")
                    progress.update()
                    continue
            
            if not times:
                raise ValueError("No valid speed measurements obtained")
            
            results = {
                "mean_ms": np.mean(times),
                "std_ms": np.std(times),
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "median_ms": np.median(times),
                "valid_trials": len(times),
                "total_trials": num_trials
            }
            
            print(f"   ✅ Speed benchmark completed: {len(times)}/{num_trials} valid trials")
            return results
            
        except Exception as e:
            error_msg = f"Speed benchmarking failed: {e}"
            self.errors.append(error_msg)
            print(f"   ❌ {error_msg}")
            return {
                "mean_ms": float('inf'),
                "error": str(e)
            }
    
    def run_comprehensive_test(self, model_name: str = "EdgeFormer") -> Dict[str, Any]:
        """Run comprehensive quantization test with fallback options."""
        print(f"\n🚀 Comprehensive EdgeFormer INT4 Test: {model_name}")
        print("=" * 60)
        
        test_results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "edgeformer_available": EDGEFORMER_AVAILABLE
        }
        
        try:
            # Create or load model
            if EDGEFORMER_AVAILABLE:
                print("📦 Creating EdgeFormer model...")
                try:
                    config = EdgeFormerConfig()
                    original_model = EdgeFormer(config).to(self.device)
                    print("   ✅ EdgeFormer model created successfully")
                except Exception as e:
                    print(f"   ❌ EdgeFormer creation failed: {e}")
                    original_model = self._create_fallback_model()
            else:
                print("📦 Creating fallback transformer model...")
                original_model = self._create_fallback_model()
            
            # Measure original model
            print("\n📊 Analyzing original model...")
            original_size = self.measure_model_size(original_model)
            original_speed = self.benchmark_inference_speed(original_model, num_trials=5)
            
            test_results["original"] = {
                "size_mb": original_size,
                "speed": original_speed
            }
            
            print(f"   Original model size: {original_size:.2f} MB")
            print(f"   Original inference: {original_speed.get('mean_ms', 'N/A'):.2f} ms")
            
            # Apply quantization
            print("\n🔧 Applying EdgeFormer INT4 quantization...")
            try:
                if EDGEFORMER_AVAILABLE:
                    quantized_model = quantize_model(original_model, bits=4)
                    print("   ✅ EdgeFormer quantization successful")
                    quantization_method = "EdgeFormer INT4"
                else:
                    # Simulate quantization for demonstration
                    quantized_model = self._simulate_quantization(original_model)
                    print("   📋 Simulated quantization applied")
                    quantization_method = "Simulated INT4"
                
                # Measure quantized model
                print("\n📊 Analyzing quantized model...")
                quantized_size = self.measure_model_size(quantized_model)
                quantized_speed = self.benchmark_inference_speed(quantized_model, num_trials=5)
                
                test_results["quantized"] = {
                    "size_mb": quantized_size,
                    "speed": quantized_speed,
                    "method": quantization_method
                }
                
                print(f"   Quantized model size: {quantized_size:.2f} MB")
                print(f"   Quantized inference: {quantized_speed.get('mean_ms', 'N/A'):.2f} ms")
                
                # Calculate compression metrics
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
                speedup = original_speed.get('mean_ms', 1) / quantized_speed.get('mean_ms', 1) if quantized_speed.get('mean_ms', 0) > 0 else 0
                memory_savings = ((original_size - quantized_size) / original_size * 100) if original_size > 0 else 0
                
                test_results["metrics"] = {
                    "compression_ratio": compression_ratio,
                    "speedup": speedup,
                    "memory_savings_percent": memory_savings
                }
                
                print(f"\n📈 Compression Metrics:")
                print(f"   Compression ratio: {compression_ratio:.1f}x")
                print(f"   Speedup: {speedup:.2f}x")
                print(f"   Memory savings: {memory_savings:.1f}%")
                
                # Test accuracy preservation
                print("\n🎯 Testing accuracy preservation...")
                accuracy_results = self.test_model_accuracy(
                    original_model, quantized_model, 
                    sequence_length=256, num_samples=5
                )
                
                test_results["accuracy"] = accuracy_results
                
                accuracy_loss = 100 - accuracy_results.get("similarity", 0)
                print(f"   Accuracy loss: {accuracy_loss:.3f}%")
                print(f"   Similarity score: {accuracy_results.get('similarity', 0):.2f}%")
                
                # Evaluate against EdgeFormer targets
                print(f"\n🏆 EdgeFormer Target Evaluation:")
                targets_met = []
                
                if compression_ratio >= 8.0:
                    print(f"   ✅ Compression target: {compression_ratio:.1f}x ≥ 8.0x")
                    targets_met.append("compression")
                else:
                    print(f"   ⚠️  Compression target: {compression_ratio:.1f}x < 8.0x")
                
                if accuracy_loss < 1.0:
                    print(f"   ✅ Accuracy target: {accuracy_loss:.3f}% < 1.0%")
                    targets_met.append("accuracy")
                else:
                    print(f"   ⚠️  Accuracy target: {accuracy_loss:.3f}% ≥ 1.0%")
                
                if speedup > 1.0:
                    print(f"   ✅ Performance target: {speedup:.2f}x speedup")
                    targets_met.append("performance")
                else:
                    print(f"   ⚠️  Performance: {speedup:.2f}x (may vary by hardware)")
                
                test_results["targets_met"] = targets_met
                success_rate = len(targets_met) / 3 * 100
                
                print(f"\n📊 Overall Success Rate: {success_rate:.0f}% ({len(targets_met)}/3 targets)")
                
                if success_rate >= 67:
                    print(f"   ✅ EdgeFormer performing well - ready for hardware validation")
                else:
                    print(f"   🔧 Algorithm optimization needed")
                
            except Exception as e:
                error_msg = f"Quantization process failed: {e}"
                self.errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                test_results["quantization_error"] = str(e)
                
        except Exception as e:
            error_msg = f"Test setup failed: {e}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
            test_results["setup_error"] = str(e)
        
        # Add error summary
        if self.errors:
            test_results["errors"] = self.errors
            print(f"\n⚠️  {len(self.errors)} errors encountered during testing")
        
        # Save results
        self._save_results(test_results)
        
        return test_results
    
    def _create_fallback_model(self) -> torch.nn.Module:
        """Create a fallback transformer model for testing."""
        class FallbackTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(10000, 512)
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True)
                    for _ in range(6)
                ])
                self.output = torch.nn.Linear(512, 10000)
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)
        
        return FallbackTransformer().to(self.device)
    
    def _simulate_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Simulate INT4 quantization for demonstration."""
        # Create a copy and modify some parameters to simulate compression
        import copy
        quantized = copy.deepcopy(model)
        
        # Simulate parameter reduction (this is just for demonstration)
        for param in quantized.parameters():
            if param.requires_grad:
                # Simulate quantization noise
                noise = torch.randn_like(param) * 0.01
                param.data = param.data + noise
        
        return quantized
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"edgeformer_test_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"📁 Results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a professional summary report."""
        print(f"\n📋 EdgeFormer INT4 Quantization Summary Report")
        print("=" * 60)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"📊 Test Overview:")
        print(f"   • Model: {results.get('model_name', 'Unknown')}")
        print(f"   • Device: {results.get('device', 'Unknown')}")
        print(f"   • Duration: {duration.total_seconds():.1f} seconds")
        print(f"   • EdgeFormer Available: {'✅' if results.get('edgeformer_available') else '❌'}")
        
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n🎯 Key Performance Metrics:")
            print(f"   • Compression Ratio: {metrics.get('compression_ratio', 0):.1f}x")
            print(f"   • Inference Speedup: {metrics.get('speedup', 0):.2f}x")
            print(f"   • Memory Savings: {metrics.get('memory_savings_percent', 0):.1f}%")
        
        if 'accuracy' in results:
            accuracy = results['accuracy']
            print(f"\n🎯 Accuracy Preservation:")
            print(f"   • Similarity Score: {accuracy.get('similarity', 0):.2f}%")
            print(f"   • Valid Samples: {accuracy.get('valid_samples', 0)}/{accuracy.get('total_samples', 0)}")
        
        targets_met = results.get('targets_met', [])
        print(f"\n🏆 EdgeFormer Targets:")
        print(f"   • Compression (8x+): {'✅' if 'compression' in targets_met else '❌'}")
        print(f"   • Accuracy (<1% loss): {'✅' if 'accuracy' in targets_met else '❌'}")
        print(f"   • Performance (speedup): {'✅' if 'performance' in targets_met else '❌'}")
        
        if self.errors:
            print(f"\n⚠️  Issues Encountered:")
            for i, error in enumerate(self.errors[:5], 1):  # Show first 5 errors
                print(f"   {i}. {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more")
        
        print(f"\n🚀 Recommendations:")
        if len(targets_met) >= 2:
            print(f"   • Proceed with hardware validation")
            print(f"   • Consider industry pilot programs")
        else:
            print(f"   • Optimize algorithm parameters")
            print(f"   • Review quantization strategy")
        
        print(f"   • Test on target hardware platforms")
        print(f"   • Expand model architecture coverage")


def main():
    """Run the enhanced EdgeFormer INT4 quantization test."""
    print("🔬 Enhanced EdgeFormer INT4 Quantization Test Suite")
    print("Professional testing with comprehensive error handling")
    print("=" * 60)
    
    tester = EnhancedQuantizationTester()
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test("EdgeFormer-INT4")
        
        # Generate summary report
        tester.generate_summary_report(results)
        
        print(f"\n🎉 Enhanced testing completed!")
        print(f"💡 Next: Run hardware validation with: python showcase_edgeformer_fixed.py")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()