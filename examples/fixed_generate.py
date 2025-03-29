import torch
import argparse
import os
import sys
sys.path.append('.')  # Add the current directory to path

from src.utils.model_loading import load_custom_model
from src.utils.tokenizer_utils import get_tokenizer_from_vocab

def main():
    parser = argparse.ArgumentParser(description="Simple text generation test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is", help="Prompt text")
    parser.add_argument("--max_length", type=int, default=48, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    args = parser.parse_args()
    
    # Load model
    model = load_custom_model(args.model_path)
    if model is None:
        print("Failed to load model")
        return
    
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = get_tokenizer_from_vocab(args.vocab_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids])
    
    # Generate
    print(f"Prompt: {args.prompt}")
    print("Generating with temperature:", args.temperature)
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k
            )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print("Generated text:")
        print(generated_text)
        
        # Try different temperatures
        if args.temperature == 0.8:
            print("\nTrying with temperature 0.5:")
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, 
                    max_length=args.max_length,
                    temperature=0.5,
                    top_k=args.top_k
                )
            generated_text = tokenizer.decode(output_ids[0].tolist())
            print(generated_text)
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()