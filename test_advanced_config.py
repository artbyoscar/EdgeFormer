#!/usr/bin/env python3
"""
Test EdgeFormer Advanced Configuration System
Run this to verify the new configuration system works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic configuration system functionality"""
    print("🔧 TESTING EDGEFORMER ADVANCED CONFIGURATION SYSTEM")
    print("=" * 60)
    
    try:
        # Import the new configuration system
        from src.config.edgeformer_config import (
            EdgeFormerDeploymentConfig,
            get_medical_grade_config,
            get_automotive_config,
            get_raspberry_pi_config,
            list_available_presets
        )
        
        print("✅ Configuration system imported successfully!")
        
        # Test 1: List available presets
        print("\n📋 Available Presets:")
        presets = list_available_presets()
        for name, description in presets.items():
            print(f"  • {name}: {description}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Make sure you saved the file to: src/config/edgeformer_config.py")
        return False
    except Exception as e:
        print(f"❌ Configuration system error: {e}")
        return False

def test_preset_configurations():
    """Test all preset configurations"""
    print("\n🧪 TESTING PRESET CONFIGURATIONS")
    print("=" * 45)
    
    try:
        from src.config.edgeformer_config import EdgeFormerDeploymentConfig
        
        # Test each preset
        preset_names = ["medical_grade", "automotive_adas", "balanced_production", 
                       "maximum_compression", "raspberry_pi_optimized"]
        
        for preset_name in preset_names:
            print(f"\n🔧 Testing {preset_name} preset...")
            
            config = EdgeFormerDeploymentConfig.from_preset(preset_name)
            
            # Get quantization parameters (for your existing quantization system)
            quant_params = config.get_quantization_params()
            
            print(f"  ✅ Preset loaded: {config.description}")
            print(f"  📊 Block size: {quant_params['block_size']}")
            print(f"  📊 Symmetric: {quant_params['symmetric']}")
            print(f"  📊 Skip layers: {len(quant_params['skip_layers'])} layers")
            print(f"  📊 Target accuracy loss: {config.accuracy.target_accuracy_loss}%")
            
            # Test performance estimation
            performance = config.estimate_performance(100.0)  # 100MB model
            print(f"  📈 Expected compression: {performance['compression_ratio']}x")
            print(f"  📈 Expected accuracy loss: {performance['accuracy_loss_percent']}%")
            
        return True
        
    except Exception as e:
        print(f"❌ Preset testing failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions"""
    print("\n🚀 TESTING CONVENIENCE FUNCTIONS")
    print("=" * 40)
    
    try:
        from src.config.edgeformer_config import (
            get_medical_grade_config,
            get_automotive_config,
            get_raspberry_pi_config
        )
        
        # Test medical grade
        print("🏥 Testing medical grade configuration...")
        medical_config = get_medical_grade_config()
        print(f"  ✅ Medical config: {medical_config.accuracy.target_accuracy_loss}% accuracy target")
        
        # Test automotive 
        print("🚗 Testing automotive ADAS configuration...")
        automotive_config = get_automotive_config()
        print(f"  ✅ Automotive config: {automotive_config.accuracy.target_accuracy_loss}% accuracy target")
        
        # Test Raspberry Pi
        print("🍓 Testing Raspberry Pi configuration...")
        pi_config = get_raspberry_pi_config()
        print(f"  ✅ Raspberry Pi config: {pi_config.deployment.target_hardware}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function testing failed: {e}")
        return False

def test_integration_with_existing_config():
    """Test integration with your existing config system"""
    print("\n🔗 TESTING INTEGRATION WITH EXISTING CONFIG")
    print("=" * 45)
    
    try:
        # Import your existing config
        from src.utils.config import DEFAULT_CONFIG, get_device_config
        from src.config.edgeformer_config import EdgeFormerDeploymentConfig
        
        print("✅ Successfully imported both config systems!")
        
        # Test integration
        print("🔧 Testing integration...")
        
        # Create config with your existing base config
        config = EdgeFormerDeploymentConfig.from_preset("automotive_adas", base_config=DEFAULT_CONFIG)
        
        # Get model config (should merge with your existing config)
        model_config = config.get_model_config()
        
        print(f"  ✅ Base config integrated")
        print(f"  📊 Model hidden size: {model_config['model']['hidden_size']}")
        print(f"  📊 Quantization type: {model_config['optimization']['quantization']}")
        print(f"  📊 Target hardware: {model_config['optimization']['deployment_target']}")
        
        # Test hardware detection
        device_info = get_device_config()
        print(f"  🖥️ Detected RAM: {device_info['ram_gb']:.1f}GB")
        print(f"  🖥️ Recommended quantization: {device_info['recommended_quantization']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration testing failed: {e}")
        return False

def test_quantization_params_compatibility():
    """Test that quantization params work with your existing system"""
    print("\n🔄 TESTING QUANTIZATION PARAMS COMPATIBILITY")
    print("=" * 50)
    
    try:
        from src.config.edgeformer_config import get_automotive_config
        
        # Get automotive config (your proven 0.5% accuracy result)
        config = get_automotive_config()
        quant_params = config.get_quantization_params()
        
        print("✅ Quantization parameters extracted:")
        for key, value in quant_params.items():
            print(f"  • {key}: {value}")
        
        # These params should work directly with your quantize_model function
        print("\n📝 Usage with your existing quantization:")
        print("   from utils.quantization import quantize_model")
        print("   compressed = quantize_model(model, **quant_params)")
        
        # Validate expected structure
        required_keys = ['quantization_type', 'block_size', 'symmetric', 'skip_layers']
        missing_keys = [key for key in required_keys if key not in quant_params]
        
        if not missing_keys:
            print("✅ All required quantization parameters present!")
        else:
            print(f"❌ Missing parameters: {missing_keys}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization compatibility test failed: {e}")
        return False

def test_save_load_functionality():
    """Test configuration save/load"""
    print("\n💾 TESTING SAVE/LOAD FUNCTIONALITY")
    print("=" * 35)
    
    try:
        from src.config.edgeformer_config import EdgeFormerDeploymentConfig
        
        # Create a test config
        original_config = EdgeFormerDeploymentConfig.from_preset("medical_grade")
        
        # Save it
        test_config_path = "test_config.json"
        success = original_config.save(test_config_path)
        
        if success:
            print("✅ Configuration saved successfully")
            
            # Load it back
            loaded_config = EdgeFormerDeploymentConfig.load(test_config_path)
            print("✅ Configuration loaded successfully")
            
            # Verify it's the same
            if (loaded_config.accuracy.target_accuracy_loss == original_config.accuracy.target_accuracy_loss and
                loaded_config.quantization.block_size == original_config.quantization.block_size):
                print("✅ Loaded configuration matches original!")
                
                # Clean up
                os.remove(test_config_path)
                print("🧹 Test file cleaned up")
                return True
            else:
                print("❌ Loaded configuration doesn't match original")
                return False
        else:
            print("❌ Failed to save configuration")
            return False
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 EDGEFORMER ADVANCED CONFIGURATION SYSTEM TESTS")
    print("=" * 55)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Preset Configurations", test_preset_configurations),
        ("Convenience Functions", test_convenience_functions),
        ("Existing Config Integration", test_integration_with_existing_config),
        ("Quantization Compatibility", test_quantization_params_compatibility),
        ("Save/Load Functionality", test_save_load_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 55)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 55)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Configuration system is ready!")
        print("\n🚀 NEXT STEPS:")
        print("   1. Integration: Update showcase_edgeformer.py to use new presets")
        print("   2. Testing: Test with your existing quantization system") 
        print("   3. Development: Move to Micro-Task 1B (Intelligent Model Analysis)")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()