#!/usr/bin/env python3
"""
Quick script to check if all EdgeFormer imports are working correctly
"""

import sys
from pathlib import Path

def check_project_structure():
    """Check if the project structure is correct"""
    print("🔍 Checking EdgeFormer project structure...")
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Expected files and directories
    expected_structure = {
        "src": "directory",
        "src/compression": "directory", 
        "src/compression/int4_quantization.py": "file",
        "src/compression/utils.py": "file",
        "src/adapters": "directory",
        "src/adapters/gpt_adapter.py": "file"
    }
    
    print("\n📁 Checking project structure:")
    all_good = True
    
    for path, path_type in expected_structure.items():
        full_path = current_dir / path
        if path_type == "directory":
            if full_path.is_dir():
                print(f"✅ {path}/")
            else:
                print(f"❌ {path}/ (missing directory)")
                all_good = False
        else:  # file
            if full_path.is_file():
                print(f"✅ {path}")
            else:
                print(f"❌ {path} (missing file)")
                all_good = False
    
    return all_good

def test_imports():
    """Test if imports work correctly"""
    print("\n🔧 Testing imports...")
    
    # Add project root to path
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    imports_successful = True
    
    # Test imports
    try:
        print("   Importing INT4Quantizer...", end=" ")
        from src.compression.int4_quantization import INT4Quantizer
        print("✅")
    except ImportError as e:
        print(f"❌ {e}")
        imports_successful = False
    
    try:
        print("   Importing utils...", end=" ")
        from src.compression.utils import calculate_model_size
        print("✅")
    except ImportError as e:
        print(f"❌ {e}")
        imports_successful = False
    
    try:
        print("   Testing PyTorch...", end=" ")
        import torch
        print("✅")
    except ImportError as e:
        print(f"❌ {e}")
        imports_successful = False
    
    try:
        print("   Testing PIL...", end=" ")
        from PIL import Image
        print("✅")
    except ImportError as e:
        print("⚠️  PIL not found - installing: pip install Pillow")
        imports_successful = False
    
    try:
        print("   Testing torchvision...", end=" ")
        import torchvision.transforms as transforms
        print("✅")
    except ImportError as e:
        print("⚠️  torchvision not found - installing: pip install torchvision")
        imports_successful = False
    
    return imports_successful

def main():
    print("🚀 EdgeFormer Import Verification")
    print("=" * 50)
    
    # Check structure
    structure_ok = check_project_structure()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if structure_ok and imports_ok:
        print("🎉 All checks passed! ViT adapter should work correctly.")
        print("\n🚀 Run the ViT adapter:")
        print("   python src/adapters/vit_adapter.py")
    else:
        print("❌ Some issues found. Please fix the following:")
        
        if not structure_ok:
            print("\n📁 Project Structure Issues:")
            print("   - Make sure you're in the EdgeFormer root directory")
            print("   - Ensure all required files and folders exist")
        
        if not imports_ok:
            print("\n🔧 Import Issues:")
            print("   - Install missing packages:")
            print("     pip install Pillow torchvision")
            print("   - Check that compression modules exist and are correct")
    
    print("\n💡 If you need to see your current file structure:")
    print("   ls -la src/compression/")
    print("   ls -la src/adapters/")

if __name__ == "__main__":
    main()