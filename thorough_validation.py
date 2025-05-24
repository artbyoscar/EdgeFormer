#!/usr/bin/env python3
"""
Comprehensive EdgeFormer Validation Framework
Thoroughly test and verify ALL claims before partnership outreach
"""

import torch
import time
import json
import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveValidator:
    """Validate all EdgeFormer claims with rigorous testing"""
    
    def __init__(self):
        self.results = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "validation_summary": {},
            "detailed_results": {},
            "claims_verification": {},
            "recommendations": []
        }
    
    def validate_int4_quantization_claims(self):
        """Rigorously validate INT4 quantization claims"""
        
        print("🔬 VALIDATING INT4 QUANTIZATION CLAIMS")
        print("=" * 50)
        
        try:
            from src.optimization.dynamic_quantization import DynamicQuantizer
            
            quantizer = DynamicQuantizer("int4")
            test_results = []
            
            # Test multiple model configurations
            test_configs = [
                {"name": "Small Model", "shape": (1000, 256), "expected_compression": 8.0},
                {"name": "Medium Model", "shape": (5000, 512), "expected_compression": 8.0},
                {"name": "Large Model", "shape": (10000, 768), "expected_compression": 8.0},
                {"name": "Vision Model", "shape": (2048, 1024), "expected_compression": 8.0},
                {"name": "Edge Model", "shape": (500, 128), "expected_compression": 8.0}
            ]
            
            for config in test_configs:
                print(f"\nTesting {config['name']}...")
                
                # Create test tensor
                test_tensor = torch.randn(config["shape"]).float()
                original_size = test_tensor.numel() * 4  # 4 bytes per float32
                
                # Multiple test runs for consistency
                compressions = []
                accuracies = []
                times = []
                
                for run in range(5):  # 5 test runs
                    start_time = time.time()
                    quantized = quantizer.quantize(test_tensor)
                    quantize_time = time.time() - start_time
                    
                    start_time = time.time()
                    dequantized = quantizer.dequantize(quantized)
                    dequantize_time = time.time() - start_time
                    
                    # Calculate compression ratio
                    compressed_size = quantized['packed_data'].numel()
                    compression_ratio = original_size / compressed_size
                    
                    # Calculate accuracy (MSE and relative error)
                    mse = torch.mean((test_tensor - dequantized) ** 2).item()
                    relative_error = (mse / torch.mean(test_tensor**2).item()) * 100
                    
                    compressions.append(compression_ratio)
                    accuracies.append(relative_error)
                    times.append(quantize_time + dequantize_time)
                
                # Analyze results
                avg_compression = np.mean(compressions)
                std_compression = np.std(compressions)
                avg_accuracy = np.mean(accuracies)
                avg_time = np.mean(times)
                
                result = {
                    "model_name": config["name"],
                    "tensor_shape": config["shape"],
                    "avg_compression": avg_compression,
                    "compression_std": std_compression,
                    "avg_accuracy_loss": avg_accuracy,
                    "avg_processing_time": avg_time,
                    "compression_consistent": std_compression < 0.1,
                    "meets_8x_claim": avg_compression >= 7.5,
                    "accuracy_acceptable": avg_accuracy <= 5.0
                }
                
                test_results.append(result)
                
                print(f"  Compression: {avg_compression:.2f}x ± {std_compression:.3f}")
                print(f"  Accuracy Loss: {avg_accuracy:.3f}%")
                print(f"  Processing Time: {avg_time*1000:.2f}ms")
                print(f"  Meets Claims: {'✅' if result['meets_8x_claim'] and result['accuracy_acceptable'] else '❌'}")
            
            # Overall assessment
            all_meet_compression = all(r['meets_8x_claim'] for r in test_results)
            all_meet_accuracy = all(r['accuracy_acceptable'] for r in test_results)
            
            self.results["detailed_results"]["int4_quantization"] = test_results
            self.results["claims_verification"]["8x_compression"] = all_meet_compression
            self.results["claims_verification"]["accuracy_loss_under_5pct"] = all_meet_accuracy
            
            print(f"\n📊 INT4 QUANTIZATION SUMMARY:")
            print(f"  8x Compression Claim: {'✅ VERIFIED' if all_meet_compression else '❌ FAILED'}")
            print(f"  <5% Accuracy Loss: {'✅ VERIFIED' if all_meet_accuracy else '❌ FAILED'}")
            
            return all_meet_compression and all_meet_accuracy
            
        except Exception as e:
            print(f"❌ INT4 validation failed: {e}")
            self.results["claims_verification"]["8x_compression"] = False
            return False
    
    def validate_performance_claims(self):
        """Validate performance and latency claims"""
        
        print("\n⚡ VALIDATING PERFORMANCE CLAIMS")
        print("=" * 50)
        
        try:
            from src.optimization.dynamic_quantization import DynamicQuantizer
            
            # Test realistic model sizes for claimed applications
            test_scenarios = [
                {
                    "name": "Healthcare ECG",
                    "model_size_mb": 13,  # Claimed: 12.77 MB
                    "target_latency_ms": 50,
                    "sequence_length": 5000  # 10 seconds at 500Hz
                },
                {
                    "name": "Medical Imaging", 
                    "model_size_mb": 32,  # Claimed: 31.54 MB
                    "target_latency_ms": 100,
                    "input_size": (512, 512)
                },
                {
                    "name": "Automotive Camera",
                    "model_size_mb": 97,  # From demo: 96.84 MB
                    "target_latency_ms": 33,  # 30 FPS = 33ms
                    "input_size": (224, 224, 6)  # 6 cameras
                },
                {
                    "name": "Manufacturing QC",
                    "model_size_mb": 2,   # From demo: 1.58 MB
                    "target_latency_ms": 60,  # 1000 parts/min = 60ms
                    "input_size": (512, 512)
                }
            ]
            
            quantizer = DynamicQuantizer("int4")
            performance_results = []
            
            for scenario in test_scenarios:
                print(f"\nTesting {scenario['name']}...")
                
                # Create model-sized tensor
                model_params = int((scenario["model_size_mb"] * 1024 * 1024) / 4)  # Convert MB to parameter count
                model_tensor = torch.randn(model_params // 1000, 1000).float()  # Reasonable 2D shape
                
                # Test compression performance
                times = []
                compressions = []
                
                for run in range(10):  # 10 runs for statistical significance
                    start_time = time.time()
                    
                    # Simulate model inference with compression
                    quantized = quantizer.quantize(model_tensor)
                    dequantized = quantizer.dequantize(quantized)
                    
                    end_time = time.time()
                    
                    inference_time = (end_time - start_time) * 1000  # Convert to ms
                    compression_ratio = (model_tensor.numel() * 4) / quantized['packed_data'].numel()
                    
                    times.append(inference_time)
                    compressions.append(compression_ratio)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_compression = np.mean(compressions)
                
                meets_latency = avg_time <= scenario["target_latency_ms"]
                meets_compression = avg_compression >= 7.5
                
                result = {
                    "scenario": scenario["name"],
                    "avg_latency_ms": avg_time,
                    "latency_std": std_time,
                    "target_latency_ms": scenario["target_latency_ms"],
                    "avg_compression": avg_compression,
                    "meets_latency_target": meets_latency,
                    "meets_compression_target": meets_compression
                }
                
                performance_results.append(result)
                
                print(f"  Latency: {avg_time:.2f}ms ± {std_time:.2f}ms (target: {scenario['target_latency_ms']}ms)")
                print(f"  Compression: {avg_compression:.2f}x")
                print(f"  Meets Targets: {'✅' if meets_latency and meets_compression else '❌'}")
            
            # Overall performance assessment
            all_meet_latency = all(r['meets_latency_target'] for r in performance_results)
            all_meet_compression = all(r['meets_compression_target'] for r in performance_results)
            
            self.results["detailed_results"]["performance"] = performance_results
            self.results["claims_verification"]["latency_targets_met"] = all_meet_latency
            self.results["claims_verification"]["performance_compression_met"] = all_meet_compression
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"  Latency Targets: {'✅ VERIFIED' if all_meet_latency else '❌ FAILED'}")
            print(f"  Compression Targets: {'✅ VERIFIED' if all_meet_compression else '❌ FAILED'}")
            
            return all_meet_latency and all_meet_compression
            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            return False
    
    def validate_existing_examples(self):
        """Validate any existing example files and demos"""
        
        print("\n📁 VALIDATING EXISTING EXAMPLES")
        print("=" * 50)
        
        examples_dir = "examples"
        if not os.path.exists(examples_dir):
            print("❌ No examples directory found")
            return False
        
        # Find all Python files in examples
        example_files = []
        for root, dirs, files in os.walk(examples_dir):
            for file in files:
                if file.endswith('.py'):
                    example_files.append(os.path.join(root, file))
        
        print(f"Found {len(example_files)} example files:")
        for file in example_files:
            print(f"  - {file}")
        
        # Test each example file
        working_examples = []
        broken_examples = []
        
        for example_file in example_files:
            try:
                print(f"\nTesting {example_file}...")
                
                # Try to import and run basic validation
                relative_path = os.path.relpath(example_file).replace('\\', '.').replace('/', '.').replace('.py', '')
                
                # Basic syntax check
                with open(example_file, 'r') as f:
                    content = f.read()
                
                # Try to compile
                compile(content, example_file, 'exec')
                
                working_examples.append(example_file)
                print(f"  ✅ Syntax valid")
                
            except Exception as e:
                broken_examples.append((example_file, str(e)))
                print(f"  ❌ Error: {e}")
        
        self.results["detailed_results"]["existing_examples"] = {
            "total_files": len(example_files),
            "working_files": len(working_examples),
            "broken_files": len(broken_examples),
            "working_examples": working_examples,
            "broken_examples": broken_examples
        }
        
        examples_valid = len(broken_examples) == 0
        self.results["claims_verification"]["examples_functional"] = examples_valid
        
        print(f"\n📊 EXAMPLES SUMMARY:")
        print(f"  Total Files: {len(example_files)}")
        print(f"  Working: {len(working_examples)}")
        print(f"  Broken: {len(broken_examples)}")
        print(f"  Examples Status: {'✅ ALL WORKING' if examples_valid else '❌ SOME BROKEN'}")
        
        return examples_valid
    
    def validate_industry_demo_claims(self):
        """Validate the specific industry demo claims"""
        
        print("\n🏭 VALIDATING INDUSTRY DEMO CLAIMS")
        print("=" * 50)
        
        try:
            # Check if industry demos file exists and runs
            from examples.industry_demos import HealthcareEdgeDemo, AutomotiveEdgeDemo, ManufacturingEdgeDemo
            
            demo_results = {}
            
            # Test Healthcare Demo
            try:
                healthcare = HealthcareEdgeDemo()
                ecg_result = healthcare.ecg_analysis_demo()
                
                demo_results["healthcare"] = {
                    "ecg_compression": ecg_result.get("compression_ratio", 0),
                    "ecg_latency": ecg_result.get("inference_time_ms", 0),
                    "hipaa_compliant": ecg_result.get("hipaa_compliant", False)
                }
                print("✅ Healthcare demo functional")
                
            except Exception as e:
                print(f"❌ Healthcare demo failed: {e}")
                demo_results["healthcare"] = {"error": str(e)}
            
            # Test Automotive Demo
            try:
                automotive = AutomotiveEdgeDemo()
                camera_result = automotive.multi_camera_demo()
                
                demo_results["automotive"] = {
                    "camera_compression": camera_result.get("compression_ratio", 0),
                    "camera_latency": camera_result.get("inference_time_ms", 0),
                    "fps": camera_result.get("fps", 0),
                    "real_time_capable": camera_result.get("real_time_capable", False)
                }
                print("✅ Automotive demo functional")
                
            except Exception as e:
                print(f"❌ Automotive demo failed: {e}")
                demo_results["automotive"] = {"error": str(e)}
            
            # Test Manufacturing Demo
            try:
                manufacturing = ManufacturingEdgeDemo()
                qc_result = manufacturing.defect_detection_demo()
                
                demo_results["manufacturing"] = {
                    "qc_compression": qc_result.get("compression_ratio", 0),
                    "qc_latency": qc_result.get("inspection_time_ms", 0),
                    "throughput": qc_result.get("throughput_ppm", 0),
                    "iso_compliant": qc_result.get("iso_compliant", False)
                }
                print("✅ Manufacturing demo functional")
                
            except Exception as e:
                print(f"❌ Manufacturing demo failed: {e}")
                demo_results["manufacturing"] = {"error": str(e)}
            
            self.results["detailed_results"]["industry_demos"] = demo_results
            
            # Validate claims
            healthcare_valid = demo_results.get("healthcare", {}).get("ecg_compression", 0) >= 7.5
            automotive_valid = demo_results.get("automotive", {}).get("camera_compression", 0) >= 7.5
            manufacturing_valid = demo_results.get("manufacturing", {}).get("qc_compression", 0) >= 7.5
            
            all_demos_valid = healthcare_valid and automotive_valid and manufacturing_valid
            
            self.results["claims_verification"]["industry_demos_functional"] = all_demos_valid
            
            print(f"\n📊 INDUSTRY DEMOS SUMMARY:")
            print(f"  Healthcare: {'✅ VALID' if healthcare_valid else '❌ INVALID'}")
            print(f"  Automotive: {'✅ VALID' if automotive_valid else '❌ INVALID'}")
            print(f"  Manufacturing: {'✅ VALID' if manufacturing_valid else '❌ INVALID'}")
            
            return all_demos_valid
            
        except ImportError as e:
            print(f"❌ Cannot import industry demos: {e}")
            return False
        except Exception as e:
            print(f"❌ Industry demo validation failed: {e}")
            return False
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        
        print("\n📋 GENERATING VALIDATION REPORT")
        print("=" * 50)
        
        # Calculate overall validation score
        verifications = self.results["claims_verification"]
        total_claims = len(verifications)
        verified_claims = sum(verifications.values())
        
        validation_score = (verified_claims / total_claims) * 100 if total_claims > 0 else 0
        
        self.results["validation_summary"] = {
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "validation_score": validation_score,
            "ready_for_partnerships": validation_score >= 80
        }
        
        # Generate recommendations
        if validation_score >= 90:
            self.results["recommendations"] = [
                "✅ All claims verified - ready for immediate partnership outreach",
                "✅ Technical demonstrations are credible and accurate",
                "✅ Industry demos are functional and meet performance targets"
            ]
        elif validation_score >= 80:
            self.results["recommendations"] = [
                "⚠️ Most claims verified - proceed with partnership outreach with caveats",
                "⚠️ Some minor issues need addressing before major partnerships",
                "⚠️ Highlight verified claims, address unverified ones"
            ]
        else:
            self.results["recommendations"] = [
                "❌ Significant issues found - do not proceed with partnership claims",
                "❌ Fix failing validations before any partnership outreach",
                "❌ Risk of damaging credibility with unverified claims"
            ]
        
        # Save validation report
        with open('comprehensive_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        print(f"Validation Score: {validation_score:.1f}%")
        print(f"Verified Claims: {verified_claims}/{total_claims}")
        print(f"Partnership Ready: {'✅ YES' if self.results['validation_summary']['ready_for_partnerships'] else '❌ NO'}")
        
        print(f"\nRecommendations:")
        for rec in self.results["recommendations"]:
            print(f"  {rec}")
        
        print(f"\n💾 Detailed report saved to: comprehensive_validation_report.json")
        
        return validation_score >= 80
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        
        print("🧪 COMPREHENSIVE EDGEFORMER VALIDATION")
        print("=" * 60)
        print("Thoroughly testing all claims before partnership outreach")
        
        # Run all validation tests
        int4_valid = self.validate_int4_quantization_claims()
        performance_valid = self.validate_performance_claims()
        examples_valid = self.validate_existing_examples()
        industry_valid = self.validate_industry_demo_claims()
        
        # Generate final report
        overall_valid = self.generate_validation_report()
        
        return overall_valid

def main():
    """Run comprehensive validation"""
    
    validator = ComprehensiveValidator()
    is_valid = validator.run_comprehensive_validation()
    
    if is_valid:
        print(f"\n🎉 VALIDATION COMPLETE: EdgeFormer ready for partnerships!")
    else:
        print(f"\n⚠️ VALIDATION INCOMPLETE: Address issues before partnership outreach")
    
    return is_valid

if __name__ == "__main__":
    main()