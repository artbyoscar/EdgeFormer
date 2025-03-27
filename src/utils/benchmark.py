import time
import torch

def benchmark(model, input_ids, runs=10, device=None):
    """Benchmark model inference speed."""
    if device is None:
        device = next(model.parameters()).device
    
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Warmup
    for _ in range(3):
        _ = model.generate(input_ids, max_length=20)
    
    # Measure time
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    for _ in range(runs):
        _ = model.generate(input_ids, max_length=20)
        
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / runs
    return avg_time