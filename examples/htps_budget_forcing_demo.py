import argparse
import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer, EdgeFormerConfig
from src.utils.htps_budget_manager import HTPSBudgetManager
from src.utils.text_dataset import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Demonstrate HyperTree Budget Forcing")
    parser.add_argument("--prompt", type=str, default="Solve this problem:", help="Input prompt")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens before budget enforcement")
    parser.add_argument("--extension_token", type=str, default="Wait", help="Token to insert for thinking extension")
    parser.add_argument("--extensions", type=int, default=2, help="Maximum number of thinking extensions")
    parser.add_argument("--criteria", type=str, default="balanced", choices=["speed", "quality", "balanced"], 
                        help="Path selection criteria")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--attention_type", type=str, default="mla", 
                        choices=["standard", "mla", "mla_window"], help="Attention type")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    print(f"Initializing EdgeFormer with {args.attention_type} attention...")
    
    # Create model config
    config = EdgeFormerConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,  # Change from num_layers to num_hidden_layers
        num_attention_heads=8,
        attention_type=args.attention_type,
        max_position_embeddings=2048,  # Changed from max_seq_length
        enable_budget_forcing=True,
        max_budget_tokens=args.max_tokens,
        max_thinking_extensions=args.extensions,
        extension_token=args.extension_token,
        budget_criteria=args.criteria
    )
    
    # Create or load model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = EdgeFormer.from_pretrained(args.model_path, config)
    else:
        print("Initializing new model")
        model = EdgeFormer(config)
    
    model.to(device)
    model.eval()
    
    # Set up the budget manager
    budget_manager = HTPSBudgetManager(
        max_tokens=args.max_tokens,
        extension_token=args.extension_token,
        max_extensions=args.extensions,
        device=args.device
    )
    
    # Tokenize input
    input_text = args.prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    print(f"\nPrompt: {input_text}")
    print(f"Budget: {args.max_tokens} tokens, {args.extensions} possible extensions")
    print("\nGenerating with budget enforcement...")
    
    # Generate text
    output_ids = model.generate(
        input_ids,
        max_length=2048,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        budget_manager=budget_manager,
        task_complexity=None  # Let the manager estimate complexity
    )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0])
    
    print("\nGenerated text:")
    print(output_text)
    
    print("\nBudget statistics:")
    print(f"Total tokens generated: {budget_manager.token_count}")
    print(f"Thinking extensions used: {budget_manager.extensions_used} of {args.extensions}")
    
    # Show path selection if multiple paths were considered
    if budget_manager.path_scores:
        print("\nComputation paths considered:")
        for path_id, score in budget_manager.path_scores.items():
            print(f"Path {path_id}: score {score:.4f}")

if __name__ == "__main__":
    main()