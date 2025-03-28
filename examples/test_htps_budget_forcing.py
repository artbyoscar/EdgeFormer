import argparse
import torch
import sys
import os
import time
import json
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer, EdgeFormerConfig
from src.utils.htps_budget_manager import HTPSBudgetManager
from src.utils.text_dataset import get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Test HyperTree Budget Forcing")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length to test")
    parser.add_argument("--forced_extensions", type=int, default=2, help="Number of forced thinking extensions")
    parser.add_argument("--extension_token", type=str, default="Wait", help="Token to use for thinking extension")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "obqa", "strategy"], 
                        help="Task type to test")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--attention_type", type=str, default="mla", 
                        choices=["standard", "mla", "mla_window"], help="Attention type")
    return parser.parse_args()

def get_test_prompts(task):
    """Get test prompts for a given task."""
    if task == "gsm8k":
        return [
            "Solve this math problem step by step: James has 5 apples. He buys 3 more apples and then gives 2 apples to his friend. How many apples does James have now?",
            "Solve this math problem step by step: A store sells shirts for $15 each and pants for $25 each. If I buy 3 shirts and 2 pants, how much will I spend in total?"
        ]
    elif task == "obqa":
        return [
            "Answer this question with step by step reasoning: What is the main source of energy for Earth's ecosystems?",
            "Answer this question with step by step reasoning: Why do objects float better in salt water than in fresh water?"
        ]
    else:  # strategy
        return [
            "Develop a comprehensive strategy for: Starting a small online business selling handmade crafts",
            "Develop a comprehensive strategy for: Implementing a recycling program in a large office building"
        ]

def estimate_task_complexity(task, prompt):
    """Estimate task complexity based on task type and prompt content."""
    complexities = {
        "gsm8k": 0.7,
        "obqa": 0.5,
        "strategy": 0.9
    }
    
    # Start with base complexity for the task
    complexity = complexities.get(task, 0.6)
    
    # Adjust based on prompt length
    words = prompt.split()
    if len(words) > 30:
        complexity += 0.1
    
    # Adjust based on specific keywords
    complexity_indicators = ["comprehensive", "detailed", "step by step", "analyze", "complex"]
    for indicator in complexity_indicators:
        if indicator in prompt.lower():
            complexity += 0.05
            
    return min(1.0, complexity)  # Cap at 1.0

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    print(f"Initializing EdgeFormer with {args.attention_type} attention...")
    
    # Create model config
    config = EdgeFormerConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=4,  # Change from num_layers to num_hidden_layers
        num_attention_heads=8,
        attention_type=args.attention_type,
        max_position_embeddings=args.sequence_length,  # Changed from max_seq_length
        enable_budget_forcing=True,
        max_budget_tokens=args.sequence_length // 2,
        max_thinking_extensions=args.forced_extensions,
        extension_token=args.extension_token
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
        max_tokens=args.sequence_length // 2,  # Use half the sequence length as budget
        extension_token=args.extension_token,
        max_extensions=args.forced_extensions,
        device=args.device
    )
    
    # Get test prompts
    prompts = get_test_prompts(args.task)
    
    results = []
    
    print(f"\nTesting HyperTree Budget Forcing on {args.task} task...")
    
    for prompt in tqdm(prompts):
        # Estimate task complexity
        task_complexity = estimate_task_complexity(args.task, prompt)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Test with budget forcing
        start_time = time.time()
        
        with torch.no_grad():
            output_ids_with_budget = model.generate(
                input_ids,
                max_length=args.sequence_length,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                budget_manager=budget_manager,
                task_complexity=task_complexity
            )
        
        time_with_budget = time.time() - start_time
        
        # Reset budget manager state
        budget_manager.reset()
        
        # Test without budget forcing
        start_time = time.time()
        
        with torch.no_grad():
            output_ids_without_budget = model.generate(
                input_ids,
                max_length=args.sequence_length,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
        
        time_without_budget = time.time() - start_time
        
        # Decode outputs
        output_with_budget = tokenizer.decode(output_ids_with_budget[0])
        output_without_budget = tokenizer.decode(output_ids_without_budget[0])
        
        # Calculate metrics
        tokens_with_budget = len(output_ids_with_budget[0])
        tokens_without_budget = len(output_ids_without_budget[0])
        
        # Save results
        result = {
            "prompt": prompt,
            "task": args.task,
            "estimated_complexity": task_complexity,
            "with_budget": {
                "time": time_with_budget,
                "tokens": tokens_with_budget,
                "extensions_used": budget_manager.extensions_used,
                "output_length": len(output_with_budget)
            },
            "without_budget": {
                "time": time_without_budget,
                "tokens": tokens_without_budget,
                "output_length": len(output_without_budget)
            },
            "comparison": {
                "time_ratio": time_with_budget / time_without_budget,
                "token_ratio": tokens_with_budget / tokens_without_budget
            }
        }
        
        results.append(result)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f"budget_forcing_results_{args.task}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nBudget Forcing Test Results:")
    print(f"Task: {args.task}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Forced Extensions: {args.forced_extensions}")
    
    avg_time_ratio = sum(r["comparison"]["time_ratio"] for r in results) / len(results)
    avg_token_ratio = sum(r["comparison"]["token_ratio"] for r in results) / len(results)
    avg_extensions = sum(r["with_budget"]["extensions_used"] for r in results) / len(results)
    
    print(f"Average Time Ratio (Budget/No Budget): {avg_time_ratio:.2f}")
    print(f"Average Token Ratio (Budget/No Budget): {avg_token_ratio:.2f}")
    print(f"Average Extensions Used: {avg_extensions:.2f} of {args.forced_extensions}")
    print(f"Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()