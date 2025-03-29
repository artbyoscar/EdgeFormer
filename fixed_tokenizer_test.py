import torch
import os

# Load the vocabulary
vocab_path = 'data/focused/vocab.pt'
if os.path.exists(vocab_path):
    vocab = torch.load(vocab_path)
    print(f"Vocabulary loaded, type: {type(vocab)}")
    
    # Extract the nested dictionaries
    if 'char_to_idx' in vocab and 'idx_to_char' in vocab:
        char_to_idx = vocab['char_to_idx']
        idx_to_char = vocab['idx_to_char']
        vocab_size = vocab['vocab_size']
        
        print(f"Vocabulary size: {vocab_size}")
        print(f"First 10 characters: {list(char_to_idx.items())[:10]}")
        
        # Try to encode and decode text
        test_text = "EdgeFormer is"
        print(f"\nAttempting to encode/decode: '{test_text}'")
        
        # Encode text
        encoded = [char_to_idx.get(char, vocab_size-1) for char in test_text]
        print(f"Encoded: {encoded}")
        
        # Decode text
        decoded = ''.join([idx_to_char.get(idx, '[UNK]') for idx in encoded])
        print(f"Decoded: '{decoded}'")
        print(f"Match original: {decoded == test_text}")
    else:
        print("Unexpected vocabulary structure")
else:
    print(f"Vocabulary file not found at {vocab_path}")

print("\nTokenizer test completed.")