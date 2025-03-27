import torch
from transformers import AutoTokenizer

# Import your config class directly 
from src.model.config import EdgeFormerConfig
from src.model.edgeformer import EdgeFormer

def main():
    # Create a small model configuration for testing
    config = EdgeFormerConfig(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=256,   # Smaller size for testing 
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        latent_size_factor=8,  # MLA parameter
        use_sliding_window=True,
        sliding_window_size=256,
        use_flash_attention=False,  # Disable for testing
        use_sparse_mlp=True,
        mlp_sparsity=0.8,
        quantization=None,  # No quantization for testing
    )
    
    print("Creating model...")
    model = EdgeFormer(config)
    
    # Use GPT-2 tokenizer for testing
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Move model to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Test inference
    print("Running inference...")
    prompt = "Hello, my name is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    print("Generating text...")
    output_ids = model.generate(
        input_ids,
        max_length=20,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Memory usage test
    print("\nMemory usage analysis:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    if torch.cuda.is_available():
        print(f"GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()