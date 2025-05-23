#!/usr/bin/env python3
"""
Fix GQA edge case: Update test configurations to use valid head/group combinations
The issue: 512 ÷ 12 = 42.67 (not integer) - need hidden_size divisible by num_heads
"""

import os

def fix_gqa_test_config():
    """Update GQA test to use valid configurations"""
    
    # Find the failing test file
    test_files = [
        "tests/model/test_gqa.py",
        "tests/model/test_gqa_simplified.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Checking {test_file}...")
            
            # Read the file
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Look for the problematic configuration
            if '"heads": 12, "groups": 3' in content and 'hidden_size": 512' in content:
                print(f"Found problematic config in {test_file}")
                
                # Replace with valid configuration
                old_config = '{"heads": 12, "groups": 3, "hidden_size": 512'
                new_config = '{"heads": 12, "groups": 3, "hidden_size": 768'  # 768÷12=64 ✓
                
                updated_content = content.replace(old_config, new_config)
                
                # Write back the fixed file
                with open(test_file, 'w') as f:
                    f.write(updated_content)
                
                print(f"✅ Fixed {test_file}: Updated hidden_size 512→768 for 12 heads")
                
            else:
                print(f"No problematic config found in {test_file}")

def validate_fix():
    """Run the GQA tests to validate the fix"""
    import subprocess
    
    print("\n--- Testing GQA fix ---")
    
    try:
        # Test the main GQA
        result = subprocess.run(['python', '-m', 'unittest', 'tests.model.test_gqa', '-v'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ tests.model.test_gqa: PASSED")
        else:
            print("❌ tests.model.test_gqa: FAILED")
            print(result.stdout)
            print(result.stderr)
            
        # Test the simplified GQA
        result2 = subprocess.run(['python', '-m', 'unittest', 'tests.model.test_gqa_simplified', '-v'], 
                               capture_output=True, text=True)
        
        if result2.returncode == 0:
            print("✅ tests.model.test_gqa_simplified: PASSED")
        else:
            print("❌ tests.model.test_gqa_simplified: FAILED")
            print(result2.stdout)
            
    except Exception as e:
        print(f"Error running tests: {e}")

def main():
    print("🔧 FIXING GQA EDGE CASE")
    print("=" * 30)
    
    # Fix the configuration
    fix_gqa_test_config()
    
    # Validate the fix works
    validate_fix()
    
    print("\n🎯 NEXT STEPS:")
    print("1. If tests pass, commit the fix")
    print("2. Move to enhanced simulation development")
    print("3. Start hardware research for Raspberry Pi purchase")

if __name__ == "__main__":
    main()