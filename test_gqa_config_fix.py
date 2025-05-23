import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.transformer.config import EdgeFormerConfig

def test_gqa_config_validation():
    """Test GQA configuration validation"""
    print("🔧 TESTING GQA CONFIG VALIDATION")
    print("=" * 40)
    
    # Valid configurations
    valid_configs = [
        {"num_attention_heads": 8, "num_key_value_heads": 2, "attention_type": "gqa", "hidden_size": 512},
        {"num_attention_heads": 8, "num_key_value_heads": 4, "attention_type": "gqa", "hidden_size": 512},
        {"num_attention_heads": 16, "num_key_value_heads": 4, "attention_type": "gqa", "hidden_size": 512},
    ]
    
    print("Testing valid GQA configurations:")
    for config_params in valid_configs:
        try:
            config = EdgeFormerConfig(**config_params)
            print(f"✅ Valid: {config_params['num_attention_heads']} heads, {config_params['num_key_value_heads']} groups")
        except Exception as e:
            print(f"❌ Should be valid: {e}")
    
    # Invalid configurations (should be rejected)
    invalid_configs = [
        {"num_attention_heads": 12, "num_key_value_heads": 5, "attention_type": "gqa", "hidden_size": 512, "reason": "12 not divisible by 5"},
        {"num_attention_heads": 8, "num_key_value_heads": 3, "attention_type": "gqa", "hidden_size": 512, "reason": "8 not divisible by 3"},
    ]
    
    print("\nTesting invalid GQA configurations (should be rejected):")
    for config_params in invalid_configs:
        try:
            config = EdgeFormerConfig(**{k: v for k, v in config_params.items() if k != "reason"})
            print(f"❌ Should have been rejected: {config_params['reason']}")
        except Exception as e:
            print(f"✅ Correctly rejected: {config_params['reason']}")

if __name__ == "__main__":
    test_gqa_config_validation()