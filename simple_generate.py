import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate text with the simple model")
    parser.add_argument("--model_path", type=str, default="checkpoints/simple_best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="EdgeFormer is", help="Prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    args = parser.parse_args()
    
    # Load vocabulary
    vocab_path = 'data/focused/vocab.pt'
    vocab = torch.load(vocab_path)
    vocab_size = vocab['vocab_size']
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    
    # Define the SimpleModel class (same as in simple_train.py)
    class SimpleModel(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size=128, num_layers=4):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
            self.lstm = torch.nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids):
            embeddings = self.embedding(input_ids)
            lstm_out, _ = self.lstm(embeddings)
            logits = self.lm_head(lstm_out)
            return logits
        
        def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
            self.eval()
            current_input = input_ids.clone()
            
            with torch.no_grad():
                for _ in range(max_length - input_ids.size(1)):
                    # Get prediction for next token
                    outputs = self(current_input)
                    next_token_logits = outputs[:, -1, :] / temperature
                    
                    # Apply top-k sampling
                    if top_k is not None:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the distribution
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to current input
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
            return current_input
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path)
    
    # Create model and load state dict
    model = SimpleModel(vocab_size, hidden_size=128, num_layers=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Create input tensor from prompt
    prompt_ids = torch.tensor([[char_to_idx.get(char, vocab_size-1) for char in args.prompt]])
    print(f"Encoded prompt: {prompt_ids}")
    
    # Generate text
    print(f"Generating text from prompt: '{args.prompt}'")
    generated_ids = model.generate(
        prompt_ids, 
        max_length=args.max_length, 
        temperature=args.temperature, 
        top_k=args.top_k
    )
    
    # Decode
    generated_text = ''.join([idx_to_char.get(id.item(), '[UNK]') for id in generated_ids[0]])
    print(f"Generated text:\n{generated_text}")
    
    # Try different temperatures
    print("\nTrying different temperatures:")
    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature: {temp}")
        generated_ids = model.generate(
            prompt_ids, 
            max_length=args.max_length, 
            temperature=temp, 
            top_k=args.top_k
        )
        generated_text = ''.join([idx_to_char.get(id.item(), '[UNK]') for id in generated_ids[0]])
        print(generated_text)

if __name__ == "__main__":
    main()