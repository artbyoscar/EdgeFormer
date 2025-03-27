# examples/check_config.py
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import EdgeFormerConfig
import inspect

def check_config():
    # Get the parameters accepted by EdgeFormerConfig
    sig = inspect.signature(EdgeFormerConfig.__init__)
    print("EdgeFormerConfig accepts these parameters:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}")
            if param.default is not inspect.Parameter.empty:
                print(f"    (default: {param.default})")

if __name__ == "__main__":
    check_config()