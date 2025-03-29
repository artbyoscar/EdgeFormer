import torch
import os
import sys

# Load the vocabulary
vocab_path = 'data/focused/vocab.pt'
if os.path.exists(vocab_path):
    vocab = torch.load(vocab_path)
    print(f"Vocabulary loaded, type: {type(vocab)}")
    if isinstance(vocab, dict):
        print(f"Vocabulary size: {len(vocab)}")
        print(f"First 10 items: {list(vocab.items())[:10]}")
    else:
        print(f"Vocabulary structure: {vocab}")
else:
    print(f"Vocabulary file not found at {vocab_path}")

# Try to manually encode and decode text
test_text = "EdgeFormer is"

# Check what type of tokenization is used in your implementation
# Character-level tokenization
if isinstance(vocab, dict) and len(vocab) < 100:  # Small vocab suggests character tokenization
    print("Attempting character-level tokenization:")
    encoded = [vocab.get(char, len(vocab)-1) for char in test_text]
    print(f"Encoded '{test_text}': {encoded}")
    
    # Try to decode
    id_to_token = {v: k for k, v in vocab.items()}
    decoded = ''.join([id_to_token.get(id, '[UNK]') for id in encoded])
    print(f"Decoded: '{decoded}'")
    print(f"Match original: {decoded == test_text}")

print("Tokenizer test completed.")
