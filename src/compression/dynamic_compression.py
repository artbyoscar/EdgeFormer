#!/usr/bin/env python3
"""
Dynamic Adaptive Compression for EdgeFormer
Real-time compression adjustment based on content and hardware constraints
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import psutil
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.compression.int4_quantization import INT4Quantizer

@dataclass
class HardwareProfile:
    """Hardware capability profile for adaptive compression"""
    memory_mb: float
    compute_capability: float  # 0-1 scale
    power_budget: float  # watts
    thermal_limit: float  # celsius
    latency_target_ms: float
    device_type: str  # "mobile", "edge", "cloud", "embedded"

@dataclass
class CompressionConfig:
    """Dynamic compression configuration"""
    base_precision: str = "int4"
    attention_boost: bool = True  # Boost attention precision for quality
    dynamic_precision: bool = True
    content_aware: bool = True
    hardware_adaptive: bool = True
    quality_target: float = 0.99  # Target quality retention (0-1)

class DynamicCompressionEngine:
    """
    Adaptive compression engine that adjusts compression in real-time
    based on content complexity, hardware constraints, and quality targets
    """
    
    def __init__(self):
        self.int4_quantizer = INT4Quantizer()
        self.compression_history = []
        self.performance_metrics = {}
        
        # Predefined hardware profiles
        self.hardware_profiles = {
            "raspberry_pi_4": HardwareProfile(
                memory_mb=4096, compute_capability=0.3, power_budget=15,
                thermal_limit=85, latency_target_ms=100, device_type="edge"
            ),
            "mobile_high_end": HardwareProfile(
                memory_mb=8192, compute_capability=0.6, power_budget=10,
                thermal_limit=45, latency_target_ms=50, device_type="mobile"
            ),
            "mobile_mid_range": HardwareProfile(
                memory_mb=4096, compute_capability=0.4, power_budget=7,
                thermal_limit=50, latency_target_ms=100, device_type="mobile"
            ),
            "embedded_mcu": HardwareProfile(
                memory_mb=512, compute_capability=0.1, power_budget=5,
                thermal_limit=70, latency_target_ms=200, device_type="embedded"
            ),
            "cloud_instance": HardwareProfile(
                memory_mb=16384, compute_capability=0.9, power_budget=150,
                thermal_limit=80, latency_target_ms=10, device_type="cloud"
            )
        }
    
    def analyze_content_complexity(self, input_tensor: torch.Tensor) -> float:
        """Analyze input content complexity to guide compression decisions"""
        if len(input_tensor.shape) < 2:
            return 0.5  # Default complexity
        
        # Calculate various complexity metrics
        complexity_metrics = []
        
        # 1. Variance-based complexity
        variance = torch.var(input_tensor.float()).item()
        normalized_variance = min(variance / 10.0, 1.0)  # Normalize to 0-1
        complexity_metrics.append(normalized_variance)
        
        # 2. Entropy-based complexity (approximation)
        flattened = input_tensor.flatten().float()
        if len(flattened) > 1:
            # Simple entropy approximation using histogram
            hist = torch.histc(flattened, bins=50)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -torch.sum(probs * torch.log2(probs + 1e-8)).item()
            normalized_entropy = min(entropy / 6.0, 1.0)  # Normalize
            complexity_metrics.append(normalized_entropy)
        
        # 3. Range-based complexity
        value_range = (input_tensor.max() - input_tensor.min()).item()
        normalized_range = min(value_range / 10.0, 1.0)
        complexity_metrics.append(normalized_range)
        
        # 4. Gradient-based complexity (if available)
        if input_tensor.requires_grad and input_tensor.grad is not None:
            grad_norm = torch.norm(input_tensor.grad).item()
            normalized_grad = min(grad_norm / 5.0, 1.0)
            complexity_metrics.append(normalized_grad)
        
        # Aggregate complexity score
        complexity = np.mean(complexity_metrics)
        return float(complexity)
    
    def get_current_hardware_state(self) -> Dict[str, float]:
        """Monitor current hardware state for adaptive decisions"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                cpu_temp = 50.0  # Default
                if temps and 'cpu_thermal' in temps:
                    cpu_temp = temps['cpu_thermal'][0].current
                elif temps and 'coretemp' in temps:
                    cpu_temp = temps['coretemp'][0].current
            except:
                cpu_temp = 50.0  # Default safe temperature
            
            # Battery (if available)
            try:
                battery = psutil.sensors_battery()
                battery_percent = battery.percent if battery else 100.0
            except:
                battery_percent = 100.0
            
            return {
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "temperature": cpu_temp,
                "battery_percent": battery_percent / 100.0
            }
        except Exception:
            # Return safe defaults if monitoring fails
            return {
                "memory_usage": 0.5,
                "cpu_usage": 0.5,
                "temperature": 50.0,
                "battery_percent": 1.0
            }
    
    def calculate_adaptive_precision(self, layer_name: str, layer_tensor: torch.Tensor,
                                   hardware_profile: HardwareProfile,
                                   content_complexity: float,
                                   config: CompressionConfig) -> str:
        """Calculate optimal precision for a layer based on multiple factors"""
        
        # Base precision from config
        base_precision = config.base_precision
        
        # Factor 1: Layer importance (based on name/type)
        layer_importance = self.get_layer_importance(layer_name)
        
        # Factor 2: Content complexity
        complexity_factor = content_complexity
        
        # Factor 3: Hardware constraints
        hardware_factor = self.get_hardware_factor(hardware_profile)
        
        # Factor 4: Quality target
        quality_factor = config.quality_target
        
        # Decision logic for precision
        precision_score = (
            layer_importance * 0.3 +
            complexity_factor * 0.3 +
            hardware_factor * 0.2 +
            quality_factor * 0.2
        )
        
        # Map score to precision
        if precision_score > 0.8:
            return "fp16"  # High precision for critical content/layers
        elif precision_score > 0.6:
            return "int8"  # Medium precision
        elif precision_score > 0.4:
            return "int4"  # Standard precision
        else:
            return "int4"  # Always at least int4
    
    def get_layer_importance(self, layer_name: str) -> float:
        """Determine layer importance score (0-1)"""
        layer_name_lower = layer_name.lower()
        
        # Critical layers (high importance)
        if any(x in layer_name_lower for x in ['attention', 'query', 'key', 'value']):
            return 0.9
        elif any(x in layer_name_lower for x in ['head', 'classifier', 'output']):
            return 0.8
        elif 'embedding' in layer_name_lower:
            return 0.7
        elif any(x in layer_name_lower for x in ['norm', 'layernorm']):
            return 0.6
        else:
            return 0.5  # Standard layers
    
    def get_hardware_factor(self, hardware_profile: HardwareProfile) -> float:
        """Calculate hardware constraint factor (0-1, higher = more constraints)"""
        constraints = []
        
        # Memory constraint
        if hardware_profile.memory_mb < 2048:
            constraints.append(0.8)  # High constraint
        elif hardware_profile.memory_mb < 4096:
            constraints.append(0.6)  # Medium constraint
        else:
            constraints.append(0.3)  # Low constraint
        
        # Compute constraint
        if hardware_profile.compute_capability < 0.3:
            constraints.append(0.8)
        elif hardware_profile.compute_capability < 0.6:
            constraints.append(0.5)
        else:
            constraints.append(0.2)
        
        # Power constraint
        if hardware_profile.power_budget < 10:
            constraints.append(0.8)
        elif hardware_profile.power_budget < 50:
            constraints.append(0.5)
        else:
            constraints.append(0.2)
        
        return np.mean(constraints)
    
    def compress_layer_adaptive(self, layer_name: str, layer_tensor: torch.Tensor,
                              hardware_profile: HardwareProfile,
                              config: CompressionConfig) -> Tuple[torch.Tensor, Dict]:
        """Compress a layer using adaptive precision"""
        
        # Analyze content complexity
        complexity = self.analyze_content_complexity(layer_tensor)
        
        # Calculate optimal precision
        precision = self.calculate_adaptive_precision(
            layer_name, layer_tensor, hardware_profile, complexity, config
        )
        
        # Perform quantization based on precision
        start_time = time.time()
        
        if precision == "int4":
            quantized, scale, zero_point = self.int4_quantizer.quantize_tensor(layer_tensor)
            compression_ratio = 8.0
            metadata = {
                "precision": precision,
                "scale": scale,
                "zero_point": zero_point,
                "compression": compression_ratio
            }
        elif precision == "int8":
            scale = layer_tensor.abs().max() / 127.0 if layer_tensor.numel() > 0 else 1.0
            quantized = torch.round(layer_tensor / scale).clamp(-128, 127).to(torch.int8)
            compression_ratio = 4.0
            metadata = {
                "precision": precision,
                "scale": scale,
                "compression": compression_ratio
            }
        else:  # fp16
            quantized = layer_tensor.to(torch.float16)
            compression_ratio = 2.0
            metadata = {
                "precision": precision,
                "compression": compression_ratio
            }
        
        compression_time = time.time() - start_time
        
        # Calculate quality metrics
        if precision == "int4":
            dequantized = self.int4_quantizer.dequantize_tensor(quantized, scale, zero_point)
        elif precision == "int8":
            dequantized = quantized.float() * metadata["scale"]
        else:
            dequantized = quantized.float()
        
        accuracy_loss = torch.mean(torch.abs(layer_tensor - dequantized)).item() * 100
        
        # Update metadata
        metadata.update({
            "complexity": complexity,
            "accuracy_loss": accuracy_loss,
            "compression_time_ms": compression_time * 1000,
            "layer_importance": self.get_layer_importance(layer_name),
            "hardware_factor": self.get_hardware_factor(hardware_profile)
        })
        
        return quantized, metadata
    
    def compress_model_adaptive(self, model: nn.Module, 
                               hardware_profile: HardwareProfile,
                               config: CompressionConfig = None) -> Dict:
        """Compress entire model using adaptive strategies"""
        
        if config is None:
            config = CompressionConfig()
        
        print(f"\n🧠 ADAPTIVE COMPRESSION FOR {hardware_profile.device_type.upper()}")
        print("=" * 70)
        print(f"Hardware Profile: {hardware_profile.memory_mb}MB RAM, "
              f"{hardware_profile.compute_capability:.1f} compute, "
              f"{hardware_profile.power_budget}W power")
        print(f"Quality Target: {config.quality_target:.1%}")
        
        # Monitor hardware state
        hardware_state = self.get_current_hardware_state()
        print(f"Current State: {hardware_state['memory_usage']:.1%} memory, "
              f"{hardware_state['cpu_usage']:.1%} CPU, "
              f"{hardware_state['temperature']:.1f}°C")
        
        results = {
            "hardware_profile": hardware_profile.device_type,
            "config": config,
            "total_layers": 0,
            "adaptive_layers": 0,
            "precision_distribution": {"int4": 0, "int8": 0, "fp16": 0},
            "compression_details": [],
            "overall_compression": 0.0,
            "avg_accuracy_loss": 0.0,
            "total_compression_time": 0.0,
            "hardware_state": hardware_state
        }
        
        total_original_size = 0
        total_compressed_size = 0
        total_accuracy_loss = 0.0
        total_compression_time = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) >= 2:
                results["total_layers"] += 1
                
                print(f"\n🔄 {name}")
                print(f"   Shape: {list(param.shape)}")
                
                try:
                    # Apply adaptive compression
                    quantized, metadata = self.compress_layer_adaptive(
                        name, param, hardware_profile, config
                    )
                    
                    # Update statistics
                    precision = metadata["precision"]
                    results["precision_distribution"][precision] += 1
                    results["adaptive_layers"] += 1
                    
                    # Calculate sizes
                    original_size = param.numel() * 4  # FP32 baseline
                    if precision == "int4":
                        compressed_size = param.numel() * 0.5
                    elif precision == "int8":
                        compressed_size = param.numel() * 1
                    else:  # fp16
                        compressed_size = param.numel() * 2
                    
                    total_original_size += original_size
                    total_compressed_size += compressed_size
                    total_accuracy_loss += metadata["accuracy_loss"]
                    total_compression_time += metadata["compression_time_ms"]
                    
                    # Store detailed results
                    layer_result = {
                        "layer": name,
                        "precision": precision,
                        "compression": metadata["compression"],
                        "accuracy_loss": metadata["accuracy_loss"],
                        "complexity": metadata["complexity"],
                        "layer_importance": metadata["layer_importance"],
                        "compression_time_ms": metadata["compression_time_ms"]
                    }
                    results["compression_details"].append(layer_result)
                    
                    print(f"   Precision: {precision}")
                    print(f"   Compression: {metadata['compression']:.1f}x")
                    print(f"   Accuracy Loss: {metadata['accuracy_loss']:.3f}%")
                    print(f"   Complexity: {metadata['complexity']:.3f}")
                    print(f"   Time: {metadata['compression_time_ms']:.2f}ms")
                    
                except Exception as e:
                    print(f"   ❌ Adaptive compression failed: {str(e)}")
                    continue
        
        # Calculate overall metrics
        if results["adaptive_layers"] > 0:
            results["overall_compression"] = total_original_size / total_compressed_size
            results["avg_accuracy_loss"] = total_accuracy_loss / results["adaptive_layers"]
            results["total_compression_time"] = total_compression_time
        
        # Print summary
        print(f"\n📊 ADAPTIVE COMPRESSION SUMMARY:")
        print(f"   Layers processed: {results['adaptive_layers']}/{results['total_layers']}")
        print(f"   Overall compression: {results['overall_compression']:.1f}x")
        print(f"   Average accuracy loss: {results['avg_accuracy_loss']:.3f}%")
        print(f"   Total compression time: {results['total_compression_time']:.1f}ms")
        print(f"   Precision distribution:")
        for precision, count in results["precision_distribution"].items():
            if count > 0:
                print(f"     {precision}: {count} layers")
        
        return results
    
    def optimize_for_runtime_constraints(self, model: nn.Module,
                                       latency_target_ms: float,
                                       memory_limit_mb: float,
                                       quality_threshold: float = 0.95) -> Dict:
        """Optimize compression for specific runtime constraints"""
        
        print(f"\n⚡ RUNTIME-CONSTRAINED OPTIMIZATION")
        print("=" * 50)
        print(f"Latency Target: {latency_target_ms}ms")
        print(f"Memory Limit: {memory_limit_mb}MB")
        print(f"Quality Threshold: {quality_threshold:.1%}")
        
        # Create custom hardware profile based on constraints
        custom_profile = HardwareProfile(
            memory_mb=memory_limit_mb,
            compute_capability=0.5,  # Assume medium capability
            power_budget=20,  # Assume moderate power budget
            thermal_limit=70,
            latency_target_ms=latency_target_ms,
            device_type="custom"
        )
        
        # Create aggressive config for tight constraints
        config = CompressionConfig(
            base_precision="int4",
            dynamic_precision=True,
            content_aware=True,
            hardware_adaptive=True,
            quality_target=quality_threshold
        )
        
        # Apply adaptive compression
        results = self.compress_model_adaptive(model, custom_profile, config)
        
        # Verify constraints are met
        estimated_memory = results["overall_compression"] 
        estimated_latency = results["total_compression_time"]
        quality_retention = 1.0 - (results["avg_accuracy_loss"] / 100.0)
        
        constraint_check = {
            "memory_constraint_met": estimated_memory <= memory_limit_mb,
            "latency_constraint_met": estimated_latency <= latency_target_ms,
            "quality_constraint_met": quality_retention >= quality_threshold,
            "estimated_memory_mb": estimated_memory,
            "estimated_latency_ms": estimated_latency,
            "quality_retention": quality_retention
        }
        
        results["constraint_verification"] = constraint_check
        
        print(f"\n✅ CONSTRAINT VERIFICATION:")
        for constraint, met in constraint_check.items():
            if isinstance(met, bool):
                status = "✅" if met else "❌"
                print(f"   {constraint}: {status}")
            else:
                print(f"   {constraint}: {met}")
        
        return results

def create_industry_profiles():
    """Create industry-specific hardware profiles"""
    return {
        "healthcare_mobile": HardwareProfile(
            memory_mb=6144, compute_capability=0.5, power_budget=12,
            thermal_limit=40, latency_target_ms=50, device_type="medical_mobile"
        ),
        "automotive_ecu": HardwareProfile(
            memory_mb=2048, compute_capability=0.4, power_budget=25,
            thermal_limit=85, latency_target_ms=33, device_type="automotive"  # 30 FPS
        ),
        "manufacturing_edge": HardwareProfile(
            memory_mb=8192, compute_capability=0.6, power_budget=50,
            thermal_limit=70, latency_target_ms=20, device_type="industrial"
        ),
        "iot_sensor": HardwareProfile(
            memory_mb=256, compute_capability=0.2, power_budget=2,
            thermal_limit=60, latency_target_ms=500, device_type="iot"
        ),
        "drone_compute": HardwareProfile(
            memory_mb=1024, compute_capability=0.4, power_budget=8,
            thermal_limit=60, latency_target_ms=100, device_type="drone"
        )
    }

def test_dynamic_compression():
    """Test dynamic compression with different scenarios"""
    print("🚀 TESTING DYNAMIC ADAPTIVE COMPRESSION")
    print("=" * 70)
    
    # Create test model
    class TestTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 512)
            self.pos_embedding = nn.Parameter(torch.randn(100, 512))
            self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
            self.norm1 = nn.LayerNorm(512)
            self.ffn = nn.Sequential(
                nn.Linear(512, 2048),
                nn.GELU(),
                nn.Linear(2048, 512)
            )
            self.norm2 = nn.LayerNorm(512)
            self.head = nn.Linear(512, 1000)
        
        def forward(self, x):
            x = self.embedding(x) + self.pos_embedding[:x.size(1)]
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return self.head(x)
    
    model = TestTransformer()
    engine = DynamicCompressionEngine()
    
    # Test different hardware profiles
    profiles = {
        "Mobile High-End": engine.hardware_profiles["mobile_high_end"],
        "Raspberry Pi 4": engine.hardware_profiles["raspberry_pi_4"],
        "Embedded MCU": engine.hardware_profiles["embedded_mcu"]
    }
    
    industry_profiles = create_industry_profiles()
    profiles.update({
        "Healthcare Mobile": industry_profiles["healthcare_mobile"],
        "Automotive ECU": industry_profiles["automotive_ecu"],
        "IoT Sensor": industry_profiles["iot_sensor"]
    })
    
    results_summary = []
    
    for profile_name, profile in profiles.items():
        print(f"\n{'='*70}")
        print(f"TESTING: {profile_name}")
        print(f"{'='*70}")
        
        try:
            results = engine.compress_model_adaptive(model, profile)
            results_summary.append({
                "profile": profile_name,
                "compression": results["overall_compression"],
                "accuracy_loss": results["avg_accuracy_loss"],
                "precision_dist": results["precision_distribution"]
            })
        except Exception as e:
            print(f"❌ Failed for {profile_name}: {str(e)}")
    
    # Test runtime-constrained optimization
    print(f"\n{'='*70}")
    print("TESTING RUNTIME-CONSTRAINED OPTIMIZATION")
    print(f"{'='*70}")
    
    constraint_tests = [
        {"latency": 50, "memory": 1024, "quality": 0.95},  # Aggressive
        {"latency": 100, "memory": 2048, "quality": 0.98}, # Balanced
        {"latency": 200, "memory": 4096, "quality": 0.99}  # Conservative
    ]
    
    for i, constraints in enumerate(constraint_tests):
        print(f"\n--- Constraint Test {i+1} ---")
        try:
            constraint_results = engine.optimize_for_runtime_constraints(
                model, 
                constraints["latency"], 
                constraints["memory"],
                constraints["quality"]
            )
            results_summary.append({
                "profile": f"Constrained-{i+1}",
                "compression": constraint_results["overall_compression"],
                "accuracy_loss": constraint_results["avg_accuracy_loss"],
                "constraints_met": all(constraint_results["constraint_verification"][k] 
                                     for k in ["memory_constraint_met", "latency_constraint_met", "quality_constraint_met"])
            })
        except Exception as e:
            print(f"❌ Constraint test {i+1} failed: {str(e)}")
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 DYNAMIC COMPRESSION TEST SUMMARY")
    print(f"{'='*70}")
    
    for result in results_summary:
        print(f"\n📊 {result['profile']}:")
        print(f"   Compression: {result.get('compression', 'N/A')}")
        if 'accuracy_loss' in result:
            print(f"   Accuracy Loss: {result['accuracy_loss']:.3f}%")
        if 'constraints_met' in result:
            status = "✅" if result['constraints_met'] else "❌"
            print(f"   Constraints Met: {status}")
    
    print("\n✅ Dynamic adaptive compression testing complete!")
    print("🚀 Ready for real-time optimization across all hardware platforms!")

if __name__ == "__main__":
    test_dynamic_compression()
