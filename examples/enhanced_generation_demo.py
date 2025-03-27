# examples/enhanced_generation_demo.py
import argparse
import torch
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('edgeformer')

def load_model(model_path, vocab_path=None, device="cpu", attention_type="standard"):
    """
    Load a trained EdgeFormer model.
    
    Args:
        model_path: Path to the model checkpoint
        vocab_path: Path to the vocabulary file (optional)
        device: Device to load the model on
        attention_type: Type of attention to use
        
    Returns:
        Loaded model
    """
    # Load checkpoint
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check what type of checkpoint we have
    if 'model_state_dict' in checkpoint:
        # Full checkpoint with optimizer state
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', None)
    else:
        # Just the model state dict
        state_dict = checkpoint
        config = None
    
    # If we don't have config in the checkpoint, try to create a basic one
    if config is None:
        from src.model.config import EdgeFormerConfig
        logger.warning("Config not found in checkpoint, using default values")
        
        # Load vocabulary if provided
        vocab_size = 256  # Default
        if vocab_path and os.path.exists(vocab_path):
            vocab_info = torch.load(vocab_path)
            vocab_size = vocab_info.get('vocab_size', 256)
            logger.info(f"Using vocabulary size from {vocab_path}: {vocab_size}")
        
        config = EdgeFormerConfig(
            vocab_size=vocab_size,
            hidden_size=256,  # Default values
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512
        )
    
    # Create model with the config
    model = EdgeFormer(config)
    
    # Set attention type
    if hasattr(model, 'set_attention_type'):
        model.set_attention_type(attention_type)
    elif hasattr(model, 'attention_type'):
        model.attention_type = attention_type
    logger.info(f"Using attention type: {attention_type}")
    
    # Load vocabulary if provided
    if vocab_path and os.path.exists(vocab_path):
        vocab_info = torch.load(vocab_path)
        model.char_to_idx = vocab_info.get('char_to_idx', {})
        model.idx_to_char = vocab_info.get('idx_to_char', {})
        logger.info(f"Loaded vocabulary with {len(model.char_to_idx)} tokens")
    
    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def generate_text(model, prompt, max_length=100, temperature=0.8, top_k=50):
    """
    Generate text using the model.
    
    Args:
        model: EdgeFormer model
        prompt: Text prompt to start generation
        max_length: Maximum length to generate
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        
    Returns:
        Generated text
    """
    logger.info(f"Generating text with prompt: '{prompt}'")
    
    # Check if model has generate method
    if not hasattr(model, 'generate'):
        logger.error("Model does not have generate method. Using manual generation.")
        return manual_generate(model, prompt, max_length, temperature, top_k)
    
    try:
        return model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
    except Exception as e:
        logger.error(f"Error in model.generate: {e}")
        logger.info("Falling back to manual generation")
        return manual_generate(model, prompt, max_length, temperature, top_k)

def manual_generate(model, prompt, max_length=100, temperature=0.8, top_k=50):
    """
    Manually generate text when the model's generate method fails.
    
    Args:
        model: EdgeFormer model
        prompt: Text prompt to start generation
        max_length: Maximum length to generate
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        
    Returns:
        Generated text
    """
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    if hasattr(model, 'char_to_idx'):
        # Use character-level tokenization
        char_to_idx = model.char_to_idx
        idx_to_char = model.idx_to_char
        
        tokens = [char_to_idx.get(c, 0) for c in prompt]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        # Fallback to simple ASCII encoding
        tokens = [ord(c) % model.config.vocab_size for c in prompt]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        idx_to_char = {i: chr(i) for i in range(128)}  # Simple ASCII mapping
    
    # Store the original prompt
    generated_text = prompt
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids)
            
            # Get the next token logits (last position, batch 0)
            next_token_logits = outputs[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][-1]
                next_token_logits[next_token_logits < indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the sampled token to the input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Add the sampled token to the generated text
            if next_token.item() in idx_to_char:
                generated_text += idx_to_char[next_token.item()]
            else:
                generated_text += ' '  # Fallback for unknown tokens
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text with trained EdgeFormer model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run generation on")
    parser.add_argument("--attention_type", type=str, default="standard", 
                      choices=["standard", "mla", "mla_window"], 
                      help="Attention mechanism to use")
    args = parser.parse_args()
    
    # Set default vocab path if not provided
    if args.vocab_path is None:
        default_vocab_path = os.path.join("data", "vocab.pt")
        if os.path.exists(default_vocab_path):
            args.vocab_path = default_vocab_path
            logger.info(f"Using default vocabulary path: {default_vocab_path}")
    
    # Load model
    model = load_model(
        args.model_path, 
        args.vocab_path, 
        args.device,
        args.attention_type
    )
    
    # Generate text
    generated_text = generate_text(
        model,
        args.prompt,
        args.max_length,
        args.temperature,
        args.top_k
    )
    
    if generated_text:
        print("\nGenerated Text:")
        print("-" * 40)
        print(generated_text)
        print("-" * 40)

if __name__ == "__main__":
    main()