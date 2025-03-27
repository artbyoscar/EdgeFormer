# examples/check_gpu_support.py
import sys
import os
import platform
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f" {text} ".center(60, "="))
    print("="*60)

def check_pytorch_installation():
    print_header("CHECKING PYTORCH INSTALLATION")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA is available")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available")
        
        # Check for ROCm support
        has_hip = hasattr(torch, 'hip')
        print(f"ROCm/HIP support in torch: {'Yes' if has_hip else 'No'}")
        
        if has_hip and torch.hip.is_available():
            print("ROCm/HIP is available and working")
            try:
                print(f"Device: {torch.hip.get_device_name(0)}")
            except:
                print("Could not get ROCm device name")
        elif has_hip:
            print("ROCm/HIP is supported but no compatible device was found")
    
    except ImportError:
        print("PyTorch is not installed")
        return False
    
    return True

def check_amd_gpu():
    print_header("CHECKING FOR AMD GPU")
    
    system = platform.system()
    if system == "Windows":
        try:
            output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
            print("Detected GPUs:")
            for line in output.split('\n'):
                if line.strip() and "name" not in line.lower():
                    print(f"  - {line.strip()}")
                    if "amd" in line.lower() or "radeon" in line.lower():
                        print("    (AMD GPU detected)")
        except Exception as e:
            print(f"Error detecting GPU: {e}")
    
    elif system == "Linux":
        try:
            # Try lspci command
            output = subprocess.check_output("lspci | grep -i 'vga\\|3d\\|display'", shell=True).decode()
            print("Detected GPUs:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"  - {line.strip()}")
                    if "amd" in line.lower() or "radeon" in line.lower():
                        print("    (AMD GPU detected)")
        except Exception as e:
            print(f"Error detecting GPU: {e}")

def check_rocm_installation():
    print_header("CHECKING FOR ROCM INSTALLATION")
    
    system = platform.system()
    if system == "Linux":
        try:
            # Check for ROCm installation
            output = subprocess.check_output("rocm-smi --showproductname", shell=True, stderr=subprocess.STDOUT).decode()
            print("ROCm is installed and the following GPUs are detected:")
            print(output)
            return True
        except subprocess.CalledProcessError:
            print("rocm-smi command failed - ROCm may not be installed or configured properly")
        except FileNotFoundError:
            print("rocm-smi command not found - ROCm is not installed")
    else:
        print(f"ROCm is primarily supported on Linux, not {system}")
        print("For Windows, AMD GPUs can be used with DirectML, but not directly with ROCm")
    
    return False

def print_recommendations():
    print_header("RECOMMENDATIONS")
    
    system = platform.system()
    if system == "Windows":
        print("On Windows with AMD GPU:")
        print("1. ROCm is not directly supported on Windows")
        print("2. Options for AMD GPU acceleration:")
        print("   a. Use DirectML backend for PyTorch (experimental)")
        print("   b. Consider using WSL2 with ROCm")
        print("   c. For your project, try CPU optimization first")
    elif system == "Linux":
        print("On Linux with AMD GPU:")
        print("1. Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html")
        print("2. Install PyTorch with ROCm support:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6")
    
    print("\nFor your EdgeFormer project:")
    print("1. Focus on optimizing the Multi-Head Latent Attention")
    print("2. Implement INT8 quantization")
    print("3. Benchmark memory usage with different context lengths")
    print("4. Consider implementing a sliding window mechanism")

def main():
    print("\nEdgeFormer GPU Compatibility Check")
    print("----------------------------------")
    
    # Check system information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Check for PyTorch installation
    check_pytorch_installation()
    
    # Check for AMD GPU
    check_amd_gpu()
    
    # Check for ROCm installation (Linux only)
    if platform.system() == "Linux":
        check_rocm_installation()
    
    # Print recommendations
    print_recommendations()

if __name__ == "__main__":
    main()