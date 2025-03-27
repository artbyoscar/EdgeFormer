import torch
try:
    import torch_directml
    dml = torch_directml.device()
    print(f"DirectML device found: {dml}")
    
    # Create a simple tensor on DirectML
    x = torch.randn(1000, 1000, device=dml)
    y = torch.randn(1000, 1000, device=dml)
    z = torch.matmul(x, y)
    print(f"Matrix multiplication successful on DirectML")
    print(f"Result shape: {z.shape}")
except ImportError:
    print("torch_directml not found. Please install it with: pip install torch-directml")