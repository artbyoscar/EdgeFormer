# Create a new file: examples/test_tokenizer.py
import torch
import argparse
from src.utils.text_dataset import SimpleTokenizer

def main():
    parser = argparse.ArgumentParser(description="Test tokenizer encode/decode")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--text", type=str, default="EdgeFormer is a transformer model.", help="Text to tokenize")
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(args.vocab_path)
    
    # Print vocabulary
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"First 10 tokens: {list(tokenizer.vocab.keys())[:10]}")
    
    # Test encode
    token_ids = tokenizer.encode(args.text)
    print(f"Encoded '{args.text}':")
    print(f"Token IDs: {token_ids}")
    
    # Print token mapping
    print("Token mapping:")
    for i, token_id in enumerate(token_ids):
        token = tokenizer.decode([token_id])
        print(f"  {token_id} -> '{token}'")
    
    # Test decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: '{decoded_text}'")
    print(f"Match original: {decoded_text == args.text}")

if __name__ == "__main__":
    main()