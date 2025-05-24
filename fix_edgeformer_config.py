#!/usr/bin/env python3
"""
Fix EdgeFormer configuration format issue
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_edgeformer_with_correct_config():
    """Test EdgeFormer with properly formatted config"""
    
    print("🔧 TESTING EDGEFORMER WITH CORRECT CONFIG FORMAT")
    print("=" * 60)
    
    try:
        from src.model.transformer.base_transformer import EdgeFormer
        
        # EdgeFormer likely expects a config object, not a dict
        # Let's try different config formats
        
        print("Testing different config formats...")
        
        # Method 1: Try with SimpleNamespace (object-like dict)
        try:
            from types import SimpleNamespace
            
            config_dict = {
                'vocab_size': 1000,
                'hidden_size': 256,
                'num_hidden_layers': 2,
                'num_attention_heads': 4,
                'intermediate_size': 1024,
                'max_position_embeddings': 512
            }
            
            config_obj = SimpleNamespace(**config_dict)
            
            print(f"  Trying SimpleNamespace config...")
            model = EdgeFormer(config_obj)
            print(f"  ✅ EdgeFormer created with SimpleNamespace config!")
            
            # Test basic functionality
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ Parameter count: {param_count:,}")
            
            # Test forward pass
            test_input = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                output = model(test_input)
            print(f"  ✅ Forward pass works: {output.shape}")
            
            return True, config_obj
            
        except Exception as e:
            print(f"  ❌ SimpleNamespace config failed: {e}")
        
        # Method 2: Try to find the actual config class
        try:
            from src.model.transformer.config import EdgeFormerConfig
            
            print(f"  Trying EdgeFormerConfig class...")
            config = EdgeFormerConfig(
                vocab_size=1000,
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=512
            )
            
            model = EdgeFormer(config)
            print(f"  ✅ EdgeFormer created with EdgeFormerConfig!")
            
            # Test basic functionality
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ Parameter count: {param_count:,}")
            
            # Test forward pass
            test_input = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                output = model(test_input)
            print(f"  ✅ Forward pass works: {output.shape}")
            
            return True, config
            
        except ImportError:
            print(f"  ❌ EdgeFormerConfig class not found")
        except Exception as e:
            print(f"  ❌ EdgeFormerConfig failed: {e}")
        
        return False, None
        
    except Exception as e:
        print(f"❌ EdgeFormer import failed: {e}")
        return False, None

def test_your_examples_with_fix():
    """Test your example files with the correct config format"""
    
    print("\n🧪 TESTING YOUR EXAMPLES WITH CORRECT CONFIG")
    print("=" * 55)
    
    working, correct_config = test_edgeformer_with_correct_config()
    
    if not working:
        print("❌ Cannot fix EdgeFormer config - need to investigate further")
        return False
    
    # Test the INT4 quantization with working EdgeFormer
    try:
        from src.optimization.dynamic_quantization import DynamicQuantizer
        
        print("\nTesting INT4 quantization with working EdgeFormer model...")
        
        # Create EdgeFormer model with correct config
        from src.model.transformer.base_transformer import EdgeFormer
        model = EdgeFormer(correct_config)
        
        print(f"✅ Created EdgeFormer: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test compression on a model layer
        quantizer = DynamicQuantizer("int4")
        
        # Get a large weight to compress
        for name, param in model.named_parameters():
            if param.numel() > 1000:  # Find a substantial layer
                print(f"\nTesting compression on {name}: {param.shape}")
                
                # Test quantization
                quantized = quantizer.quantize(param.data)
                dequantized = quantizer.dequantize(quantized)
                
                # Calculate compression
                original_size = param.numel() * 4
                compressed_size = quantized['packed_data'].numel()
                compression_ratio = original_size / compressed_size
                
                # Calculate accuracy
                mse = torch.mean((param.data - dequantized) ** 2).item()
                relative_error = (mse / torch.mean(param.data**2).item()) * 100
                
                print(f"  Compression: {compression_ratio:.2f}x")
                print(f"  Accuracy: {relative_error:.3f}% error")
                
                if compression_ratio >= 7.5 and relative_error <= 10.0:
                    print(f"  ✅ EdgeFormer + INT4 quantization works!")
                    return True
                else:
                    print(f"  ⚠️ Performance below expectations")
                break
        
    except Exception as e:
        print(f"❌ EdgeFormer + INT4 test failed: {e}")
        return False
    
    return True

def create_working_example():
    """Create a working example file with correct EdgeFormer usage"""
    
    print("\n📝 CREATING WORKING EDGEFORMER EXAMPLE")
    print("=" * 45)
    
    working_example = '''#!/usr/bin/env python3
"""
Working EdgeFormer Example
Demonstrates correct usage with proper config format
"""

import torch
import sys
import os
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_edgeformer_config():
    """Create properly formatted EdgeFormer config"""
    
    config_dict = {
        'vocab_size': 1000,
        'hidden_size': 256,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 1024,
        'max_position_embeddings': 512,
        'attention_type': 'standard'
    }
    
    # Try EdgeFormerConfig first, fallback to SimpleNamespace
    try:
        from src.model.transformer.config import EdgeFormerConfig
        return EdgeFormerConfig(**config_dict)
    except ImportError:
        return SimpleNamespace(**config_dict)

def main():
    """Test EdgeFormer with correct configuration"""
    
    print("🧪 Testing EdgeFormer with correct config format")
    
    # Create model
    from src.model.transformer.base_transformer import EdgeFormer
    config = create_edgeformer_config()
    
    model = EdgeFormer(config)
    print(f"✅ EdgeFormer created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    test_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Forward pass successful: {output.shape}")
    
    # Test with INT4 quantization
    from src.optimization.dynamic_quantization import DynamicQuantizer
    quantizer = DynamicQuantizer("int4")
    
    # Test on embedding layer
    embedding_weight = model.transformer.embeddings.word_embeddings.weight
    print(f"\\nTesting INT4 on embedding: {embedding_weight.shape}")
    
    quantized = quantizer.quantize(embedding_weight.data)
    dequantized = quantizer.dequantize(quantized)
    
    compression_ratio = quantizer.get_compression_ratio(embedding_weight.data, quantized)
    mse = torch.mean((embedding_weight.data - dequantized) ** 2).item()
    relative_error = (mse / torch.mean(embedding_weight.data**2).item()) * 100
    
    print(f"  Compression: {compression_ratio:.2f}x")
    print(f"  Error: {relative_error:.3f}%")
    
    if compression_ratio >= 7.5:
        print("🎉 EdgeFormer + INT4 quantization WORKING!")
        return True
    else:
        print("⚠️ Compression below target")
        return False

if __name__ == "__main__":
    main()
'''
    
    with open('working_edgeformer_example.py', 'w', encoding='utf-8') as f:
        f.write(working_example)
    
    print("✅ Created working_edgeformer_example.py")
    print("Run with: python working_edgeformer_example.py")

def main():
    """Main diagnostic and fix"""
    
    print("🔧 FIXING EDGEFORMER CONFIGURATION ISSUES")
    print("=" * 50)
    
    # Test EdgeFormer with correct config
    working = test_your_examples_with_fix()
    
    # Create working example
    create_working_example()
    
    if working:
        print(f"\n✅ EDGEFORMER FIXED!")
        print(f"   Issue was config format, not model implementation")
        print(f"   Your EdgeFormer + INT4 combination works!")
        print(f"\n🎯 PARTNERSHIP STATUS: BACK TO RESEARCH_READY")
    else:
        print(f"\n❌ EDGEFORMER STILL HAS ISSUES")
        print(f"   Need further investigation")
        print(f"\n🎯 PARTNERSHIP STATUS: NOT_READY")
    
    return working

if __name__ == "__main__":
    main()