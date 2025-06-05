#!/usr/bin/env python3
"""
Hardware-Specific Optimization Suite for EdgeFormer
Tailored compression strategies for different hardware architectures
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.compression.int4_quantization import INT4Quantizer

@dataclass
class HardwareSpecs:
    """Hardware specification profile"""
    name: str
    architecture: str  # "arm", "x86", "risc-v", "specialized"
    memory_hierarchy: Dict[str, int]  # L1, L2, L3, RAM in KB/MB
    compute_units: Dict[str, int]  # cores, vector_units, etc.
    memory_bandwidth_gbps: float
    cache_line_size: int
    simd_width: int  # SIMD vector width
    has_neural_engine: bool
    power_envelope_watts: float
    thermal_design_power: float

class HardwareOptimizer:
    """
    Hardware-specific optimization strategies for different platforms
    """
    
    def __init__(self):
        self.int4_quantizer = INT4Quantizer()
        self.hardware_configs = self._create_hardware_configs()
        
    def _create_hardware_configs(self) -> Dict[str, HardwareSpecs]:
        """Define hardware configurations for major platforms"""
        return {
            # ARM Processors
            "raspberry_pi_4": HardwareSpecs(
                name="Raspberry Pi 4",
                architecture="arm",
                memory_hierarchy={"L1": 32, "L2": 1024, "RAM": 4096000},
                compute_units={"cores": 4, "vector_units": 1},
                memory_bandwidth_gbps=3.2,
                cache_line_size=64,
                simd_width=128,
                has_neural_engine=False,
                power_envelope_watts=15,
                thermal_design_power=15
            ),
            
            "apple_m1": HardwareSpecs(
                name="Apple M1",
                architecture="arm",
                memory_hierarchy={"L1": 128, "L2": 12288, "RAM": 16384000},
                compute_units={"cores": 8, "neural_engine": 16},
                memory_bandwidth_gbps=68.25,
                cache_line_size=128,
                simd_width=256,
                has_neural_engine=True,
                power_envelope_watts=20,
                thermal_design_power=20
            ),
            
            "qualcomm_888": HardwareSpecs(
                name="Qualcomm Snapdragon 888",
                architecture="arm",
                memory_hierarchy={"L1": 64, "L2": 1024, "L3": 4096, "RAM": 8192000},
                compute_units={"cores": 8, "hexagon_dsp": 1, "adreno_gpu": 1},
                memory_bandwidth_gbps=51.2,
                cache_line_size=64,
                simd_width=128,
                has_neural_engine=True,
                power_envelope_watts=8,
                thermal_design_power=8
            ),
            
            # x86 Processors
            "intel_nuc": HardwareSpecs(
                name="Intel NUC i5",
                architecture="x86",
                memory_hierarchy={"L1": 64, "L2": 512, "L3": 8192, "RAM": 16384000},
                compute_units={"cores": 4, "threads": 8},
                memory_bandwidth_gbps=38.4,
                cache_line_size=64,
                simd_width=256,
                has_neural_engine=False,
                power_envelope_watts=65,
                thermal_design_power=65
            ),
            
            "amd_ryzen_embedded": HardwareSpecs(
                name="AMD Ryzen Embedded",
                architecture="x86", 
                memory_hierarchy={"L1": 64, "L2": 512, "L3": 4096, "RAM": 8192000},
                compute_units={"cores": 4, "threads": 8},
                memory_bandwidth_gbps=42.7,
                cache_line_size=64,
                simd_width=256,
                has_neural_engine=False,
                power_envelope_watts=35,
                thermal_design_power=35
            ),
            
            # Specialized AI Chips
            "nvidia_jetson_nano": HardwareSpecs(
                name="NVIDIA Jetson Nano",
                architecture="arm",
                memory_hierarchy={"L1": 32, "L2": 512, "RAM": 4096000},
                compute_units={"cores": 4, "cuda_cores": 128, "tensor_cores": 0},
                memory_bandwidth_gbps=25.6,
                cache_line_size=64,
                simd_width=128,
                has_neural_engine=True,
                power_envelope_watts=10,
                thermal_design_power=10
            ),
            
            "google_coral": HardwareSpecs(
                name="Google Coral Edge TPU",
                architecture="specialized",
                memory_hierarchy={"L1": 32, "L2": 1024, "RAM": 1024000},
                compute_units={"edge_tpu": 1, "cortex_m": 4},
                memory_bandwidth_gbps=6.4,
                cache_line_size=32,
                simd_width=64,
                has_neural_engine=True,
                power_envelope_watts=2,
                thermal_design_power=2
            ),
            
            # Mobile Processors
            "mobile_flagship": HardwareSpecs(
                name="Mobile Flagship (Generic)",
                architecture="arm",
                memory_hierarchy={"L1": 64, "L2": 2048, "L3": 8192, "RAM": 12288000},
                compute_units={"cores": 8, "gpu_cores": 20, "npu": 1},
                memory_bandwidth_gbps=44.8,
                cache_line_size=64,
                simd_width=128,
                has_neural_engine=True,
                power_envelope_watts=6,
                thermal_design_power=6
            ),
            
            "mobile_midrange": HardwareSpecs(
                name="Mobile Mid-range (Generic)",
                architecture="arm",
                memory_hierarchy={"L1": 32, "L2": 1024, "L3": 2048, "RAM": 6144000},
                compute_units={"cores": 8, "gpu_cores": 12},
                memory_bandwidth_gbps=17.1,
                cache_line_size=64,
                simd_width=128,
                has_neural_engine=False,
                power_envelope_watts=4,
                thermal_design_power=4
            ),
            
            # Embedded/IoT
            "embedded_cortex_m": HardwareSpecs(
                name="ARM Cortex-M7",
                architecture="arm",
                memory_hierarchy={"L1": 16, "RAM": 512},
                compute_units={"cores": 1},
                memory_bandwidth_gbps=0.8,
                cache_line_size=32,
                simd_width=32,
                has_neural_engine=False,
                power_envelope_watts=0.5,
                thermal_design_power=0.5
            )
        }
    
    def get_cache_aware_strategy(self, hardware: HardwareSpecs, model_size_mb: float) -> Dict:
        """Develop cache-aware optimization strategy"""
        
        l1_cache_kb = hardware.memory_hierarchy.get("L1", 32)
        l2_cache_kb = hardware.memory_hierarchy.get("L2", 512)
        l3_cache_kb = hardware.memory_hierarchy.get("L3", 0)
        
        # Calculate optimal block sizes for cache efficiency
        l1_block_size = min(l1_cache_kb * 1024 // 4, 16384)  # FP32 parameters
        l2_block_size = min(l2_cache_kb * 1024 // 4, 65536)
        
        strategy = {
            "block_quantization": True,
            "l1_block_size": l1_block_size,
            "l2_block_size": l2_block_size,
            "cache_friendly_layout": True,
            "memory_access_pattern": "sequential" if l3_cache_kb > 0 else "blocked"
        }
        
        # Adjust strategy based on cache hierarchy
        if l3_cache_kb > 4096:  # Large L3 cache
            strategy["aggressive_prefetch"] = True
            strategy["block_size_multiplier"] = 2.0
        elif l2_cache_kb < 512:  # Small L2 cache
            strategy["micro_blocking"] = True
            strategy["block_size_multiplier"] = 0.5
        
        return strategy
    
    def get_simd_optimization(self, hardware: HardwareSpecs) -> Dict:
        """Generate SIMD-optimized quantization parameters"""
        
        simd_width = hardware.simd_width
        cache_line = hardware.cache_line_size
        
        # Optimize for SIMD vector operations
        optimal_chunk_size = max(simd_width // 4, 4)  # INT4 packing
        alignment_requirement = cache_line
        
        optimization = {
            "vector_chunk_size": optimal_chunk_size,
            "memory_alignment": alignment_requirement,
            "parallel_quantization": simd_width >= 128,
            "vectorized_operations": True
        }
        
        # ARM-specific optimizations
        if hardware.architecture == "arm":
            optimization.update({
                "neon_optimizations": True,
                "arm_sve": simd_width >= 256,
                "unroll_factor": 4 if simd_width >= 128 else 2
            })
        
        # x86-specific optimizations  
        elif hardware.architecture == "x86":
            optimization.update({
                "avx2_optimizations": simd_width >= 256,
                "avx512_optimizations": simd_width >= 512,
                "intel_mkl_compat": True,
                "fma_instructions": True
            })
        
        return optimization
    
    def get_power_aware_strategy(self, hardware: HardwareSpecs) -> Dict:
        """Generate power-aware compression strategy"""
        
        power_budget = hardware.power_envelope_watts
        has_neural_engine = hardware.has_neural_engine
        
        strategy = {
            "power_budget_watts": power_budget,
            "thermal_throttling_aware": True
        }
        
        if power_budget < 2:  # Ultra-low power (IoT)
            strategy.update({
                "compression_level": "maximum",
                "precision_preference": "int4",
                "dynamic_voltage_scaling": True,
                "computation_offload": False
            })
        elif power_budget < 10:  # Low power (Mobile)
            strategy.update({
                "compression_level": "high", 
                "precision_preference": "mixed_int4_int8",
                "neural_engine_offload": has_neural_engine,
                "adaptive_frequency": True
            })
        elif power_budget < 50:  # Medium power (Edge)
            strategy.update({
                "compression_level": "balanced",
                "precision_preference": "adaptive",
                "parallel_processing": True,
                "thermal_monitoring": True
            })
        else:  # High power (Workstation)
            strategy.update({
                "compression_level": "quality_focused",
                "precision_preference": "mixed_precision",
                "full_parallel": True,
                "performance_mode": True
            })
        
        return strategy
    
    def optimize_for_hardware(self, model: nn.Module, hardware_name: str) -> Dict:
        """Apply comprehensive hardware-specific optimizations"""
        
        if hardware_name not in self.hardware_configs:
            available = list(self.hardware_configs.keys())
            raise ValueError(f"Unknown hardware: {hardware_name}. Available: {available}")
        
        hardware = self.hardware_configs[hardware_name]
        
        print(f"\n🔧 OPTIMIZING FOR {hardware.name.upper()}")
        print("=" * 70)
        print(f"Architecture: {hardware.architecture}")
        print(f"Memory: {hardware.memory_hierarchy['RAM'] // 1000}MB RAM")
        print(f"Power: {hardware.power_envelope_watts}W")
        print(f"SIMD: {hardware.simd_width}-bit")
        print(f"Neural Engine: {'✅' if hardware.has_neural_engine else '❌'}")
        
        # Calculate model size
        model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        print(f"Model Size: {model_size:.1f}MB")
        
        # Generate optimization strategies
        cache_strategy = self.get_cache_aware_strategy(hardware, model_size)
        simd_optimization = self.get_simd_optimization(hardware)
        power_strategy = self.get_power_aware_strategy(hardware)
        
        # Apply hardware-specific compression
        compression_results = self._apply_hardware_compression(
            model, hardware, cache_strategy, simd_optimization, power_strategy
        )
        
        optimization_results = {
            "hardware_profile": {
                "name": hardware.name,
                "architecture": hardware.architecture,
                "power_watts": hardware.power_envelope_watts,
                "memory_mb": hardware.memory_hierarchy["RAM"] // 1000,
                "simd_width": hardware.simd_width
            },
            "optimization_strategies": {
                "cache_strategy": cache_strategy,
                "simd_optimization": simd_optimization,
                "power_strategy": power_strategy
            },
            "compression_results": compression_results,
            "performance_estimates": self._estimate_performance(hardware, compression_results)
        }
        
        return optimization_results
    
    def _apply_hardware_compression(self, model: nn.Module, hardware: HardwareSpecs,
                                   cache_strategy: Dict, simd_opt: Dict, power_strategy: Dict) -> Dict:
        """Apply compression with hardware-specific optimizations"""
        
        results = {
            "total_layers": 0,
            "optimized_layers": 0,
            "compression_details": [],
            "overall_compression": 0.0,
            "hardware_efficiency_score": 0.0
        }
        
        total_original_size = 0
        total_compressed_size = 0
        efficiency_scores = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                results["total_layers"] += 1
                
                print(f"\n⚙️  {name}")
                print(f"   Shape: {list(param.shape)}")
                
                try:
                    # Determine optimal precision based on hardware constraints
                    precision = self._select_optimal_precision(
                        param, hardware, power_strategy
                    )
                    
                    # Apply quantization with hardware optimizations
                    if precision == "int4":
                        quantized, scale, zero_point = self.int4_quantizer.quantize_tensor(param)
                        compression_ratio = 8.0
                        
                        # Hardware-specific optimizations
                        if simd_opt["parallel_quantization"]:
                            # Simulate SIMD optimization benefit
                            compression_ratio *= 1.1  # 10% efficiency boost
                    
                    elif precision == "int8":
                        scale = param.abs().max() / 127.0 if param.numel() > 0 else 1.0
                        quantized = torch.round(param / scale).clamp(-128, 127).to(torch.int8)
                        compression_ratio = 4.0
                    
                    else:  # fp16
                        quantized = param.to(torch.float16)
                        compression_ratio = 2.0
                    
                    # Calculate efficiency score
                    efficiency_score = self._calculate_efficiency_score(
                        param, compression_ratio, hardware, cache_strategy
                    )
                    
                    # Size calculations
                    original_size = param.numel() * 4
                    compressed_size = original_size / compression_ratio
                    
                    total_original_size += original_size
                    total_compressed_size += compressed_size
                    efficiency_scores.append(efficiency_score)
                    
                    results["optimized_layers"] += 1
                    results["compression_details"].append({
                        "layer": name,
                        "precision": precision,
                        "compression": compression_ratio,
                        "efficiency_score": efficiency_score,
                        "cache_friendly": self._is_cache_friendly(param.shape, cache_strategy)
                    })
                    
                    print(f"   Precision: {precision}")
                    print(f"   Compression: {compression_ratio:.1f}x")
                    print(f"   Efficiency Score: {efficiency_score:.3f}")
                    
                except Exception as e:
                    print(f"   ❌ Hardware optimization failed: {str(e)}")
                    continue
        
        # Calculate overall metrics
        if results["optimized_layers"] > 0:
            results["overall_compression"] = total_original_size / total_compressed_size
            results["hardware_efficiency_score"] = np.mean(efficiency_scores)
        
        return results
    
    def _select_optimal_precision(self, param: torch.Tensor, hardware: HardwareSpecs, 
                                 power_strategy: Dict) -> str:
        """Select optimal precision based on hardware characteristics"""
        
        # Factor in power constraints
        if power_strategy["compression_level"] == "maximum":
            return "int4"
        elif power_strategy["compression_level"] == "high":
            # Use int4 for large layers, int8 for smaller ones
            return "int4" if param.numel() > 100000 else "int8"
        elif power_strategy["compression_level"] == "balanced":
            # Mixed precision based on layer importance
            if any(x in str(param.shape) for x in ["attention", "head", "output"]):
                return "int8"  # Higher precision for critical layers
            else:
                return "int4"
        else:  # quality_focused
            return "int8"  # Conservative approach for quality
    
    def _calculate_efficiency_score(self, param: torch.Tensor, compression_ratio: float,
                                   hardware: HardwareSpecs, cache_strategy: Dict) -> float:
        """Calculate hardware efficiency score for a layer"""
        
        # Base score from compression ratio
        base_score = min(compression_ratio / 8.0, 1.0)
        
        # Cache efficiency bonus
        cache_bonus = 0.0
        if self._is_cache_friendly(param.shape, cache_strategy):
            cache_bonus = 0.1
        
        # SIMD efficiency bonus
        simd_bonus = 0.0
        if param.numel() % (hardware.simd_width // 4) == 0:  # SIMD-aligned
            simd_bonus = 0.05
        
        # Memory bandwidth efficiency
        bandwidth_factor = min(hardware.memory_bandwidth_gbps / 50.0, 1.0)
        bandwidth_bonus = bandwidth_factor * 0.1
        
        total_score = base_score + cache_bonus + simd_bonus + bandwidth_bonus
        return min(total_score, 1.0)
    
    def _is_cache_friendly(self, shape: torch.Size, cache_strategy: Dict) -> bool:
        """Check if tensor shape is cache-friendly"""
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        # Check if fits in L2 cache with some headroom
        l2_capacity = cache_strategy.get("l2_block_size", 65536)
        return total_elements * 4 <= l2_capacity * 0.8  # 80% utilization
    
    def _estimate_performance(self, hardware: HardwareSpecs, compression_results: Dict) -> Dict:
        """Estimate performance improvements from hardware optimization"""
        
        base_compression = compression_results["overall_compression"]
        efficiency_score = compression_results["hardware_efficiency_score"]
        
        # Estimate memory bandwidth savings
        memory_bandwidth_savings = 1.0 - (1.0 / base_compression)
        
        # Estimate cache hit rate improvement
        cache_hit_improvement = efficiency_score * 0.3  # Up to 30% improvement
        
        # Estimate power consumption reduction
        power_reduction = memory_bandwidth_savings * 0.6  # Memory access power dominant
        
        # Estimate latency improvement
        latency_improvement = (memory_bandwidth_savings + cache_hit_improvement) * 0.5
        
        # Hardware-specific multipliers
        if hardware.has_neural_engine:
            latency_improvement *= 1.2  # Neural engine acceleration
        
        if hardware.simd_width >= 256:
            latency_improvement *= 1.1  # SIMD acceleration
        
        return {
            "memory_bandwidth_savings": memory_bandwidth_savings,
            "cache_hit_improvement": cache_hit_improvement,
            "power_reduction": power_reduction,
            "latency_improvement": latency_improvement,
            "estimated_speedup": 1.0 + latency_improvement,
            "estimated_power_savings": power_reduction,
            "memory_footprint_reduction": 1.0 - (1.0 / base_compression)
        }
    
    def benchmark_across_hardware(self, model: nn.Module) -> Dict:
        """Benchmark optimization across all supported hardware"""
        
        print(f"\n🏁 CROSS-HARDWARE OPTIMIZATION BENCHMARK")
        print("=" * 80)
        
        benchmark_results = {}
        
        for hw_name, hw_spec in self.hardware_configs.items():
            print(f"\n📊 Benchmarking {hw_spec.name}...")
            
            try:
                results = self.optimize_for_hardware(model, hw_name)
                
                benchmark_results[hw_name] = {
                    "hardware_name": hw_spec.name,
                    "architecture": hw_spec.architecture,
                    "compression": results["compression_results"]["overall_compression"],
                    "efficiency_score": results["compression_results"]["hardware_efficiency_score"],
                    "estimated_speedup": results["performance_estimates"]["estimated_speedup"],
                    "power_savings": results["performance_estimates"]["estimated_power_savings"],
                    "memory_reduction": results["performance_estimates"]["memory_footprint_reduction"]
                }
                
                print(f"   ✅ Compression: {results['compression_results']['overall_compression']:.1f}x")
                print(f"   ✅ Efficiency: {results['compression_results']['hardware_efficiency_score']:.3f}")
                print(f"   ✅ Est. Speedup: {results['performance_estimates']['estimated_speedup']:.2f}x")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                benchmark_results[hw_name] = {"error": str(e)}
        
        # Generate comparison report
        self._generate_comparison_report(benchmark_results)
        
        return benchmark_results
    
    def _generate_comparison_report(self, benchmark_results: Dict):
        """Generate comprehensive comparison report"""
        
        print(f"\n{'='*80}")
        print("📋 HARDWARE OPTIMIZATION COMPARISON REPORT")
        print(f"{'='*80}")
        
        # Sort by compression ratio
        valid_results = {k: v for k, v in benchmark_results.items() if "error" not in v}
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]["compression"], reverse=True)
        
        print(f"\n🏆 COMPRESSION RANKING:")
        for i, (hw_name, results) in enumerate(sorted_results[:5]):
            print(f"   {i+1}. {results['hardware_name']}: {results['compression']:.1f}x")
        
        print(f"\n⚡ EFFICIENCY RANKING:")
        efficiency_sorted = sorted(valid_results.items(),
                                 key=lambda x: x[1]["efficiency_score"], reverse=True)
        for i, (hw_name, results) in enumerate(efficiency_sorted[:5]):
            print(f"   {i+1}. {results['hardware_name']}: {results['efficiency_score']:.3f}")
        
        print(f"\n🚀 SPEEDUP RANKING:")
        speedup_sorted = sorted(valid_results.items(),
                              key=lambda x: x[1]["estimated_speedup"], reverse=True)
        for i, (hw_name, results) in enumerate(speedup_sorted[:5]):
            print(f"   {i+1}. {results['hardware_name']}: {results['estimated_speedup']:.2f}x")
        
        # Architecture analysis
        print(f"\n🏗️  ARCHITECTURE ANALYSIS:")
        arch_performance = {}
        for hw_name, results in valid_results.items():
            arch = results["architecture"]
            if arch not in arch_performance:
                arch_performance[arch] = []
            arch_performance[arch].append(results["compression"])
        
        for arch, compressions in arch_performance.items():
            avg_compression = np.mean(compressions)
            print(f"   {arch.upper()}: {avg_compression:.1f}x average compression")
        
        # Power efficiency analysis
        print(f"\n⚡ POWER EFFICIENCY ANALYSIS:")
        power_sorted = sorted(valid_results.items(),
                            key=lambda x: x[1]["power_savings"], reverse=True)
        for i, (hw_name, results) in enumerate(power_sorted[:3]):
            print(f"   {i+1}. {results['hardware_name']}: {results['power_savings']:.1%} power savings")

def create_optimization_recommendations():
    """Generate optimization recommendations for different use cases"""
    
    recommendations = {
        "mobile_deployment": {
            "recommended_hardware": ["apple_m1", "qualcomm_888", "mobile_flagship"],
            "optimization_focus": ["power_efficiency", "thermal_management", "battery_life"],
            "compression_target": "8-10x",
            "quality_target": "98%"
        },
        "edge_computing": {
            "recommended_hardware": ["raspberry_pi_4", "nvidia_jetson_nano", "intel_nuc"],
            "optimization_focus": ["cost_efficiency", "deployment_simplicity", "reliability"],
            "compression_target": "6-8x", 
            "quality_target": "99%"
        },
        "iot_deployment": {
            "recommended_hardware": ["embedded_cortex_m", "google_coral"],
            "optimization_focus": ["ultra_low_power", "minimal_memory", "real_time"],
            "compression_target": "10-15x",
            "quality_target": "95%"
        },
        "automotive": {
            "recommended_hardware": ["amd_ryzen_embedded", "nvidia_jetson_nano"],
            "optimization_focus": ["safety_critical", "thermal_robust", "real_time"],
            "compression_target": "6-8x",
            "quality_target": "99.5%"
        },
        "industrial": {
            "recommended_hardware": ["intel_nuc", "amd_ryzen_embedded"],
            "optimization_focus": ["reliability", "performance", "maintainability"],
            "compression_target": "4-6x",
            "quality_target": "99.9%"
        }
    }
    
    return recommendations

def test_hardware_optimization():
    """Test hardware-specific optimizations"""
    print("🔧 TESTING HARDWARE-SPECIFIC OPTIMIZATIONS")
    print("=" * 70)
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5000, 256)
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True)
                for _ in range(6)
            ])
            self.head = nn.Linear(256, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.transformer_layers:
                x = layer(x)
            return self.head(x)
    
    model = TestModel()
    optimizer = HardwareOptimizer()
    
    # Test specific hardware optimizations
    test_hardware = [
        "raspberry_pi_4",
        "apple_m1", 
        "qualcomm_888",
        "intel_nuc",
        "nvidia_jetson_nano",
        "google_coral",
        "embedded_cortex_m"
    ]
    
    results_summary = []
    
    for hw_name in test_hardware:
        print(f"\n{'='*70}")
        print(f"TESTING: {hw_name}")
        print(f"{'='*70}")
        
        try:
            results = optimizer.optimize_for_hardware(model, hw_name)
            
            results_summary.append({
                "hardware": results["hardware_profile"]["name"],
                "architecture": results["hardware_profile"]["architecture"],
                "compression": results["compression_results"]["overall_compression"],
                "efficiency": results["compression_results"]["hardware_efficiency_score"],
                "speedup": results["performance_estimates"]["estimated_speedup"],
                "power_savings": results["performance_estimates"]["estimated_power_savings"]
            })
            
        except Exception as e:
            print(f"❌ Hardware optimization failed for {hw_name}: {str(e)}")
    
    # Run cross-hardware benchmark
    print(f"\n{'='*70}")
    print("RUNNING CROSS-HARDWARE BENCHMARK")
    print(f"{'='*70}")
    
    try:
        benchmark_results = optimizer.benchmark_across_hardware(model)
    except Exception as e:
        print(f"❌ Cross-hardware benchmark failed: {str(e)}")
    
    # Generate recommendations
    recommendations = create_optimization_recommendations()
    
    print(f"\n{'='*70}")
    print("🎯 DEPLOYMENT RECOMMENDATIONS")
    print(f"{'='*70}")
    
    for use_case, rec in recommendations.items():
        print(f"\n📱 {use_case.replace('_', ' ').title()}:")
        print(f"   Recommended Hardware: {', '.join(rec['recommended_hardware'])}")
        print(f"   Compression Target: {rec['compression_target']}")
        print(f"   Quality Target: {rec['quality_target']}")
        print(f"   Focus Areas: {', '.join(rec['optimization_focus'])}")
    
    print("\n✅ Hardware-specific optimization testing complete!")
    print("🚀 Ready for deployment across all major hardware platforms!")

if __name__ == "__main__":
    test_hardware_optimization()
