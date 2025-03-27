# In examples/test_weight_only_quantization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.utils.quantization import weight_only_quantize_model

def test_weight_only_quantization():
    """Test weight-only quantization for EdgeFormer model."""
    # Create model configuration
    config = EdgeFormerConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_size_factor=8,
    )
    
    # Initialize model
    model = EdgeFormer(config)
    model.eval()
    
    # Create input tensors
    input_ids = torch.randint(0, config.vocab_size, (1, 512))
    attention_mask = torch.ones(1, 512)
    
    # Run reference inference
    with torch.no_grad():
        reference_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Quantize with different settings
    quantization_configs = [
        {"bits": 8, "group_size": 128, "use_symmetric": True},
        {"bits": 8, "group_size": 128, "use_symmetric": False},
        {"bits": 4, "group_size": 128, "use_symmetric": True},
        {"bits": 4, "group_size": 128, "use_symmetric": False},
        {"bits": 4, "group_size": 64, "use_symmetric": True},
        {"bits": 4, "group_size": 32, "use_symmetric": True},
    ]
    
    results = []
    
    for config_dict in quantization_configs:
        # Quantize model
        bits = config_dict["bits"]
        group_size = config_dict["group_size"]
        use_symmetric = config_dict["use_symmetric"]
        
        print(f"Testing {bits}-bit quantization with group size {group_size}, symmetric={use_symmetric}")
        
        q_model = weight_only_quantize_model(
            model, 
            bits=bits, 
            group_size=group_size, 
            use_symmetric=use_symmetric
        )
        
        # Calculate model size reduction
        orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
        q_size = sum(p.numel() * (1 if p.dtype == torch.int8 else p.element_size()) 
                    for p in q_model.parameters())
        
        size_reduction = orig_size / q_size
        
        # Run quantized inference
        with torch.no_grad():
            q_outputs = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Calculate quantization error
        orig_logits = reference_outputs["logits"]
        q_logits = q_outputs["logits"]
        
        mse = torch.mean((orig_logits - q_logits) ** 2).item()
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            orig_logits.reshape(-1), 
            q_logits.reshape(-1), 
            dim=0
        ).item()
        
        # Store results
        config_name = f"{bits}b_{group_size}g_{'sym' if use_symmetric else 'asym'}"
        results.append({
            "config": config_name,
            "bits": bits,
            "group_size": group_size,
            "symmetric": use_symmetric,
            "size_reduction": size_reduction,
            "mse": mse,
            "similarity": similarity * 100  # Convert to percentage
        })
        
        print(f"Size reduction: {size_reduction:.2f}x")
        print(f"MSE: {mse:.6f}")
        print(f"Similarity: {similarity * 100:.2f}%")
        print("-" * 50)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Size reduction vs similarity
    configs = [r["config"] for r in results]
    size_reductions = [r["size_reduction"] for r in results]
    similarities = [r["similarity"] for r in results]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(configs, size_reductions)
    plt.ylabel('Size Reduction Factor')
    plt.title('Model Size Reduction by Quantization Config')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(configs, similarities)
    plt.ylabel('Output Similarity (%)')
    plt.title('Output Quality by Quantization Config')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/weight_only_quantization.png')
    
    return results

if __name__ == "__main__":
    test_weight_only_quantization()