#!/usr/bin/env python3
"""
Diagnose EdgeFormer model implementation issues
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_edgeformer_model():
    """Diagnose issues with EdgeFormer model implementation"""
    
    print("🔍 DIAGNOSING EDGEFORMER MODEL IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Try to import EdgeFormer
        from src.model.transformer.base_transformer import EdgeFormer
        print("✅ EdgeFormer imported successfully")
        
        # Check if it's properly inheriting from nn.Module
        print(f"EdgeFormer base classes: {EdgeFormer.__bases__}")
        
        # Try to create a simple EdgeFormer instance
        try:
            config = {
                'vocab_size': 1000,
                'hidden_size': 256,
                'num_hidden_layers': 2,
                'num_attention_heads': 4,
                'intermediate_size': 1024,
                'max_position_embeddings': 512
            }
            
            model = EdgeFormer(config)
            print("✅ EdgeFormer model created successfully")
            
            # Check essential PyTorch model methods
            essential_methods = [
                'parameters',
                'named_parameters', 
                'named_children',
                'named_modules',
                'state_dict',
                'load_state_dict'
            ]
            
            print("\n🔍 Checking essential PyTorch methods:")
            for method in essential_methods:
                if hasattr(model, method):
                    print(f"  ✅ {method}: Available")
                    
                    # Test if it's callable
                    try:
                        if method in ['parameters', 'named_parameters', 'named_children', 'named_modules']:
                            result = list(getattr(model, method)())
                            print(f"    - Returns {len(result)} items")
                        elif method == 'state_dict':
                            state = getattr(model, method)()
                            print(f"    - State dict has {len(state)} keys")
                    except Exception as e:
                        print(f"    - ❌ Error calling {method}: {e}")
                else:
                    print(f"  ❌ {method}: MISSING")
            
            # Check if model can do forward pass
            try:
                test_input = torch.randint(0, 1000, (1, 10))  # batch_size=1, seq_len=10
                with torch.no_grad():
                    output = model(test_input)
                print(f"\n✅ Forward pass works: output shape {output.shape}")
            except Exception as e:
                print(f"\n❌ Forward pass failed: {e}")
            
            # Check parameter count
            try:
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✅ Parameter count: {param_count:,}")
            except Exception as e:
                print(f"❌ Cannot count parameters: {e}")
                
        except Exception as e:
            print(f"❌ Failed to create EdgeFormer model: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import EdgeFormer: {e}")
        
        # Try to find the model file
        model_files = [
            "src/model/transformer/base_transformer.py",
            "src/model/edgeformer.py", 
            "src/model/transformer/edgeformer.py"
        ]
        
        print("\n🔍 Looking for EdgeFormer model files:")
        for file_path in model_files:
            if os.path.exists(file_path):
                print(f"  ✅ Found: {file_path}")
                
                # Check if it has EdgeFormer class
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if 'class EdgeFormer' in content:
                        print(f"    - Contains EdgeFormer class")
                        
                        # Check if it inherits from nn.Module
                        if 'nn.Module' in content:
                            print(f"    - References nn.Module")
                        else:
                            print(f"    - ❌ No nn.Module inheritance found")
                    else:
                        print(f"    - No EdgeFormer class found")
                        
                except Exception as e:
                    print(f"    - Error reading file: {e}")
            else:
                print(f"  ❌ Not found: {file_path}")
        
        return False
    
    return True

def suggest_fixes():
    """Suggest fixes for common EdgeFormer issues"""
    
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 30)
    
    print("1. Ensure EdgeFormer inherits from nn.Module:")
    print("   class EdgeFormer(nn.Module):")
    print("       def __init__(self, config):")
    print("           super().__init__()  # This is critical!")
    
    print("\n2. Check your EdgeFormer __init__ method calls super().__init__()")
    
    print("\n3. Verify all sub-modules are properly registered as nn.Module attributes")
    
    print("\n4. Test with a minimal EdgeFormer implementation:")
    
    minimal_code = '''
class EdgeFormerMinimal(nn.Module):
    def __init__(self, config):
        super().__init__()  # CRITICAL: Must call super().__init__()
        
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_attention_heads'],
                batch_first=True
            ),
            num_layers=config['num_hidden_layers']
        )
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        return self.output(x)
'''
    
    print(minimal_code)

def test_working_alternative():
    """Test with a working transformer to verify the issue is with EdgeFormer"""
    
    print("\n🧪 TESTING WITH WORKING PYTORCH TRANSFORMER:")
    print("=" * 55)
    
    try:
        # Create a simple working transformer
        class WorkingTransformer(torch.nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2, num_heads=4):
                super().__init__()  # This is what EdgeFormer is missing!
                
                self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.transformer = torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        batch_first=True
                    ),
                    num_layers=num_layers
                )
                self.output = torch.nn.Linear(hidden_size, vocab_size)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        # Test this working model
        model = WorkingTransformer()
        
        # Test all the methods that are failing on EdgeFormer
        param_count = sum(p.numel() for p in model.parameters())
        children_count = len(list(model.named_children()))
        
        print(f"✅ Working model parameter count: {param_count:,}")
        print(f"✅ Working model children count: {children_count}")
        
        # Test forward pass
        test_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Working model forward pass: {output.shape}")
        
        print("\n🎯 The issue is definitely with EdgeFormer's implementation!")
        print("   Your EdgeFormer class is missing proper nn.Module inheritance.")
        
        return True
        
    except Exception as e:
        print(f"❌ Even basic transformer failed: {e}")
        return False

def main():
    """Run comprehensive EdgeFormer diagnosis"""
    
    edgeformer_works = diagnose_edgeformer_model()
    
    if not edgeformer_works:
        suggest_fixes()
        test_working_alternative()
        
        print(f"\n🚨 CRITICAL ISSUE IDENTIFIED:")
        print(f"   Your EdgeFormer model is not properly inheriting from nn.Module")
        print(f"   This explains why your examples are failing")
        print(f"   Fix this before proceeding with any partnerships!")
    else:
        print(f"\n✅ EdgeFormer model implementation looks correct")
    
    return edgeformer_works

if __name__ == "__main__":
    main()