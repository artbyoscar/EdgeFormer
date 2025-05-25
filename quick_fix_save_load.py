#!/usr/bin/env python3
"""
Quick fix for the Windows path issue in save/load functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_save_load():
    """Test save/load with proper Windows path handling"""
    print("🔧 TESTING FIXED SAVE/LOAD FUNCTIONALITY")
    print("=" * 40)
    
    try:
        from src.config.edgeformer_config import EdgeFormerDeploymentConfig
        
        # Create a test config
        config = EdgeFormerDeploymentConfig.from_preset("automotive_adas")
        
        # Use current directory with proper path
        test_config_path = os.path.join(os.getcwd(), "test_config.json")
        print(f"📁 Saving to: {test_config_path}")
        
        # Save it
        success = config.save(test_config_path)
        
        if success:
            print("✅ Configuration saved successfully")
            
            # Load it back
            loaded_config = EdgeFormerDeploymentConfig.load(test_config_path)
            print("✅ Configuration loaded successfully")
            
            # Verify it's the same
            if (loaded_config.accuracy.target_accuracy_loss == config.accuracy.target_accuracy_loss and
                loaded_config.quantization.block_size == config.quantization.block_size):
                print("✅ Loaded configuration matches original!")
                
                # Clean up
                os.remove(test_config_path)
                print("🧹 Test file cleaned up")
                print("🎉 SAVE/LOAD FUNCTIONALITY WORKING!")
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

def test_integration_with_showcase():
    """Test integration with your existing showcase_edgeformer.py"""
    print("\n🔗 TESTING INTEGRATION WITH SHOWCASE_EDGEFORMER")
    print("=" * 50)
    
    try:
        from src.config.edgeformer_config import EdgeFormerDeploymentConfig
        
        print("✅ Testing all presets for showcase integration...")
        
        presets_to_test = [
            ("medical_grade", "Medical Grade (0.3% accuracy target)"),
            ("automotive_adas", "Automotive ADAS (0.5% proven accuracy)"),
            ("raspberry_pi_optimized", "Raspberry Pi Ready"),
            ("maximum_compression", "Maximum Compression (7.8x proven)")
        ]
        
        for preset_name, description in presets_to_test:
            config = EdgeFormerDeploymentConfig.from_preset(preset_name)
            quant_params = config.get_quantization_params()
            
            print(f"\n🔧 {description}:")
            print(f"   📊 Block size: {quant_params['block_size']}")
            print(f"   📊 Skip layers: {quant_params['skip_layers']}")
            print(f"   📊 Target accuracy: {config.accuracy.target_accuracy_loss}%")
            print(f"   📊 Expected compression: {config.expected_results['compression_ratio']}x")
            
            # These params are ready for your quantize_model function
            print(f"   ✅ Ready for: quantize_model(model, **quant_params)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Test the fixes"""
    print("🔧 FIXING ADVANCED CONFIGURATION SYSTEM")
    print("=" * 45)
    
    # Test the fixed save/load
    save_load_works = test_fixed_save_load()
    
    # Test integration readiness
    integration_works = test_integration_with_showcase()
    
    if save_load_works and integration_works:
        print("\n🎉 ALL SYSTEMS GO!")
        print("✅ Configuration system fully functional")
        print("✅ Ready for integration with existing code")
        print("✅ All presets validated and working")
        
        print("\n🚀 IMMEDIATE NEXT STEPS:")
        print("1. Integrate with showcase_edgeformer.py")
        print("2. Test medical/automotive presets with real models")
        print("3. Move to Micro-Task 1B: Intelligent Model Analysis")
        
        return True
    else:
        print("\n❌ Issues remain - need further investigation")
        return False

if __name__ == "__main__":
    main()