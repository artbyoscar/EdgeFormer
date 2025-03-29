# Create a new file: examples/simple_generate.py
import torch
import argparse
from src.utils.model_loading import load_custom_model
from src.utils.text_dataset import SimpleTokenizer

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
    model.eval()
    
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(args.vocab_path)
    
    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids])
    
    # Generate
    print(f"Prompt: {args.prompt}")
    print("Generating with temperature:", args.temperature)
    
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

if __name__ == "__main__":
    main()