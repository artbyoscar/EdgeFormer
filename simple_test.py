import os
import sys
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function from examples
from examples.simple_inference import main

if __name__ == "__main__":
    main()