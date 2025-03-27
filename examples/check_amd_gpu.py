import logging
import platform
import subprocess
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_windows_gpu():
    """Check GPU information on Windows systems"""
    try:
        # Run Windows Management Instrumentation query for GPU info
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("GPU Information:\n" + result.stdout)
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to get GPU information: {e}")
        return None

def check_directml_support():
    """Check if DirectML is available for PyTorch"""
    try:
        import torch_directml
        logger.info("PyTorch DirectML is installed.")
        
        # Try to initialize DirectML device
        try:
            device = torch_directml.device()
            logger.info(f"Successfully initialized DirectML device: {device}")
            return True
        except Exception as e:
            logger.error(f"DirectML device initialization failed: {e}")
            return False
    except ImportError:
        logger.warning("PyTorch DirectML is not installed.")
        logger.info("To install, run: pip install torch-directml")
        return False

def install_directml_if_needed():
    """Install DirectML if not already installed"""
    try:
        import torch_directml
        logger.info("PyTorch DirectML is already installed.")
    except ImportError:
        logger.info("Installing PyTorch DirectML...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch-directml"],
                check=True
            )
            logger.info("PyTorch DirectML installed successfully.")
        except Exception as e:
            logger.error(f"Failed to install PyTorch DirectML: {e}")

def check_system_info():
    """Print detailed system information"""
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.warning("PyTorch not found.")
    
    # Check for AMD GPU on Windows
    if platform.system() == "Windows":
        check_windows_gpu()
    
    # Check for DirectML support
    check_directml_support()

if __name__ == "__main__":
    logger.info("Checking system configuration for AMD GPU support...")
    check_system_info()
    
    # Ask user if they want to install DirectML
    answer = input("Do you want to install PyTorch DirectML? (y/n): ")
    if answer.lower() == 'y':
        install_directml_if_needed()
        
        # Check DirectML support again after installation
        if check_directml_support():
            logger.info("DirectML is now ready to use with your AMD GPU.")
        else:
            logger.warning("DirectML setup incomplete. Please check the logs.")