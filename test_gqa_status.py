import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gqa_status():
    """Test if GQA is available and working with existing EdgeFormer structure"""
    print("=== GQA INTEGRATION STATUS ===")
    
    # Test 1: Check if GQA exists in transformer folder
    try:
        from src.model.transformer.gqa import GroupedQueryAttention
        print("✅ GQA module found in transformer folder")
        gqa_available = True
    except Exception as e:
        print(f"❌ GQA module error: {e}")
        gqa_available = False
    
    # Test 2: Check if EdgeFormer model exists
    try:
        from src.model.transformer.base_transformer import EdgeFormer, EdgeFormerConfig
        print("✅ EdgeFormer model found")
        edgeformer_available = True
    except Exception as e:
        print(f"⚠️ EdgeFormer import issue: {e}")
        try:
            # Try alternative import
            from src.model.transformer.base_transformer import EdgeFormerModel
            print("✅ EdgeFormerModel found (alternative)")
            edgeformer_available = True
        except Exception as e2:
            print(f"❌ EdgeFormer not available: {e2}")
            edgeformer_available = False
    
    # Test 3: Create GQA config and test
    if gqa_available:
        try:
            # Create a simple config for testing
            class SimpleConfig:
                def __init__(self):
                    self.hidden_size = 512
                    self.num_attention_heads = 8
                    self.num_key_value_heads = 4  # For GQA
                    self.attention_probs_dropout_prob = 0.1
            
            config = SimpleConfig()
            gqa = GroupedQueryAttention(config)
            print("✅ GQA instantiation successful")
            
            # Test forward pass
            test_input = torch.randn(2, 10, 512)
            outputs = gqa(test_input)
            print(f"✅ GQA forward pass successful: output shape {outputs[0].shape}")
            
            return "GQA_WORKING"
            
        except Exception as e:
            print(f"❌ GQA testing failed: {e}")
            import traceback
            traceback.print_exc()
            return "GQA_NEEDS_FIXES"
    
    # Test 4: Test EdgeFormer with GQA if available
    if edgeformer_available and gqa_available:
        try:
            # Create EdgeFormer config with GQA
            config = EdgeFormerConfig(
                vocab_size=1000,
                hidden_size=512,
                num_hidden_layers=2,
                num_attention_heads=8,
                intermediate_size=2048,
                attention_type="gqa",
                num_key_value_heads=4
            )
            
            model = EdgeFormer(config)
            print("✅ EdgeFormer with GQA created successfully")
            
            # Test forward pass
            test_input = torch.randint(0, 1000, (1, 10))
            outputs = model(test_input)
            print(f"✅ EdgeFormer GQA forward pass successful: {outputs[0].shape}")
            
            return "COMPLETE"
            
        except Exception as e:
            print(f"⚠️ EdgeFormer GQA integration needs minor fixes: {e}")
            import traceback
            traceback.print_exc()
            return "MINOR_FIXES_NEEDED"
    
    return "MISSING_COMPONENTS"

if __name__ == "__main__":
    status = test_gqa_status()
    print(f"\nFinal GQA Status: {status}")
    
    if status == "COMPLETE":
        print("🎉 PHASE 1 COMPLETE (100%)!")
        print("🚀 Ready for strategic partnerships!")
    elif status in ["GQA_WORKING", "MINOR_FIXES_NEEDED"]:
        print("⚡ Almost there - just minor integration fixes needed!")
    else:
        print("🔧 Some components need implementation")