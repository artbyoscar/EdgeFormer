# Create a new file: examples/model_analysis.py
import torch
import argparse
import numpy as np
from src.utils.model_loading import load_custom_model

def main():
    parser = argparse.ArgumentParser(description="Analyze model parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load model
    model = load_custom_model(args.model_path)
    
    # Print model info
    print("Model configuration:")
    for key, value in vars(model.config).items():
        print(f"  {key}: {value}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Analyze parameter statistics
    print("\nParameter statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Min: {param.min().item():.6f}")
            print(f"    Max: {param.max().item():.6f}")
            print(f"    Mean: {param.mean().item():.6f}")
            print(f"    Std: {param.std().item():.6f}")
    
    # Check for NaN or Inf values
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"Inf found in {name}")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("\nNo NaN or Inf values found in model parameters.")

if __name__ == "__main__":
    main()