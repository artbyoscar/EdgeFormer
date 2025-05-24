#!/usr/bin/env python3
"""
Test the new EdgeFormer components (ViT and Industry demos)
"""

def test_vit_implementation():
    """Test the Vision Transformer implementation"""
    
    print("🔬 TESTING EDGEFORMER VIT IMPLEMENTATION")
    print("=" * 50)
    
    try:
        # Test if we can import and run the ViT
        import sys
        sys.path.append('src/model/vision')
        
        from edgeformer_vit import EdgeFormerViT, test_edgeformer_vit, create_medical_imaging_demo
        
        print("✅ EdgeFormer ViT imported successfully")
        
        # Run the tests
        test_edgeformer_vit()
        create_medical_imaging_demo()
        
        print("✅ EdgeFormer ViT tests completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the ViT file is saved as src/model/vision/edgeformer_vit.py")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_industry_demos():
    """Test the industry demonstrations"""
    
    print("\n🏭 TESTING INDUSTRY DEMONSTRATIONS")
    print("=" * 50)
    
    try:
        # Test if we can import and run the industry demos
        import sys
        sys.path.append('examples')
        
        from industry_demos import run_comprehensive_industry_demos
        
        print("✅ Industry demos imported successfully")
        
        # Run the demonstrations
        results = run_comprehensive_industry_demos()
        
        print("✅ Industry demonstrations completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the industry demos file is saved as examples/industry_demos.py")
        return False
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def update_project_structure():
    """Update project structure and create __init__.py files"""
    
    import os
    
    # Create necessary directories
    directories = [
        'src/model/vision',
        'examples'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith('src/'):
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# EdgeFormer module\n')
                print(f"✅ Created {init_file}")
    
    print("✅ Project structure updated")

def create_integration_test():
    """Create a comprehensive integration test"""
    
    print("\n🧪 CREATING INTEGRATION TEST")
    print("=" * 40)
    
    integration_test = '''#!/usr/bin/env python3
"""
EdgeFormer Integration Test
Tests all major components together
"""

import torch
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('src')
sys.path.append('examples')

def run_integration_test():
    """Run comprehensive integration test"""
    
    print("🧪 EDGEFORMER INTEGRATION TEST")
    print("=" * 40)
    
    test_results = {
        "int4_quantization": False,
        "gqa_attention": False,
        "vision_transformer": False,
        "industry_demos": False,
        "edge_simulation": False
    }
    
    # Test 1: INT4 Quantization
    try:
        from src.optimization.dynamic_quantization import DynamicQuantizer
        quantizer = DynamicQuantizer("int4")
        test_tensor = torch.randn(10, 10)
        quantized = quantizer.quantize(test_tensor)
        dequantized = quantizer.dequantize(quantized)
        
        compression_ratio = quantizer.get_compression_ratio(test_tensor, quantized)
        if 7.5 <= compression_ratio <= 8.5:  # Allow some tolerance
            test_results["int4_quantization"] = True
            print("✅ INT4 Quantization: PASSED")
        else:
            print(f"❌ INT4 Quantization: FAILED (compression: {compression_ratio:.2f}x)")
            
    except Exception as e:
        print(f"❌ INT4 Quantization: ERROR ({e})")
    
    # Test 2: GQA Attention
    try:
        import unittest
        from tests.model.test_gqa_simplified import TestGQASimplified
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestGQASimplified)
        result = unittest.TextTestRunner(verbosity=0).run(suite)
        
        if result.wasSuccessful():
            test_results["gqa_attention"] = True
            print("✅ GQA Attention: PASSED")
        else:
            print("❌ GQA Attention: FAILED")
            
    except Exception as e:
        print(f"❌ GQA Attention: ERROR ({e})")
    
    # Test 3: Vision Transformer
    try:
        from src.model.vision.edgeformer_vit import EdgeFormerViT
        
        # Quick ViT test
        vit = EdgeFormerViT(image_size=224, num_classes=10, depth=6, compress=True)
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = vit(test_input)
        
        if output.shape == (1, 10) and torch.isfinite(output).all():
            test_results["vision_transformer"] = True
            print("✅ Vision Transformer: PASSED")
        else:
            print("❌ Vision Transformer: FAILED (invalid output)")
            
    except Exception as e:
        print(f"❌ Vision Transformer: ERROR ({e})")
    
    # Test 4: Industry Demos
    try:
        from examples.industry_demos import HealthcareEdgeDemo
        
        healthcare = HealthcareEdgeDemo()
        result = healthcare.ecg_analysis_demo()
        
        if result and result.get('compression_ratio', 0) >= 7:
            test_results["industry_demos"] = True
            print("✅ Industry Demos: PASSED")
        else:
            print("❌ Industry Demos: FAILED")
            
    except Exception as e:
        print(f"❌ Industry Demos: ERROR ({e})")
    
    # Test 5: Edge Simulation
    try:
        if os.path.exists('edge_simulation_results.json'):
            import json
            with open('edge_simulation_results.json', 'r') as f:
                sim_results = json.load(f)
            
            if sim_results and len(sim_results) >= 2:  # At least 2 devices tested
                test_results["edge_simulation"] = True
                print("✅ Edge Simulation: PASSED")
            else:
                print("❌ Edge Simulation: FAILED (insufficient results)")
        else:
            print("⚠️ Edge Simulation: SKIPPED (no results file)")
            
    except Exception as e:
        print(f"❌ Edge Simulation: ERROR ({e})")
    
    # Summary
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\\n📊 INTEGRATION TEST SUMMARY")
    print(f"=" * 40)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests >= 4:  # Allow one failure
        print("\\n🎉 INTEGRATION TEST: OVERALL PASS")
        print("EdgeFormer is ready for strategic partnerships!")
        return True
    else:
        print("\\n⚠️ INTEGRATION TEST: NEEDS ATTENTION") 
        print("Some components need fixes before partnership outreach")
        return False

if __name__ == "__main__":
    run_integration_test()
'''
    
    with open('integration_test.py', 'w') as f:
        f.write(integration_test)
    
    print("✅ Created integration_test.py")
    print("Run with: python integration_test.py")

def main():
    """Main test runner"""
    
    print("🧪 TESTING NEW EDGEFORMER COMPONENTS")
    print("=" * 50)
    
    # Update project structure first
    update_project_structure()
    
    # Test components (will fail until files are created)
    vit_ok = test_vit_implementation()
    industry_ok = test_industry_demos()
    
    # Create integration test
    create_integration_test()
    
    print(f"\\n🎯 NEXT STEPS:")
    if not vit_ok:
        print("1. Create src/model/vision/edgeformer_vit.py with the ViT code")
    if not industry_ok:
        print("2. Create examples/industry_demos.py with the industry demo code") 
    print("3. Run: python integration_test.py")
    print("4. Commit all components once tests pass")
    
    return vit_ok and industry_ok

if __name__ == "__main__":
    main()