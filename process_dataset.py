import torch
import os

# Load dataset
dataset_path = 'data/focused/text_dataset.pt'
dataset = torch.load(dataset_path)
print(f"Dataset type: {type(dataset)}")
print(f"Dataset length: {len(dataset)}")
print(f"First 20 values: {dataset[:20]}")

# Load vocabulary
vocab_path = 'data/focused/vocab.pt'
vocab = torch.load(vocab_path)
vocab_size = vocab['vocab_size']
char_to_idx = vocab['char_to_idx']
idx_to_char = vocab['idx_to_char']
print(f"Vocabulary size: {vocab_size}")

# Convert the list of integers to a tensor
# We need to determine the sequence length first
# Let's create sequences of a fixed length (e.g., 48)
seq_length = 48
sequences = []

# Create overlapping sequences
for i in range(0, len(dataset) - seq_length, 1):  # Step of 1 for maximum overlap
    sequence = dataset[i:i+seq_length]
    sequences.append(sequence)

print(f"Created {len(sequences)} sequences of length {seq_length}")

# Convert to tensor
sequences_tensor = torch.tensor(sequences, dtype=torch.long)
print(f"Sequences tensor shape: {sequences_tensor.shape}")

# Save the preprocessed dataset
torch.save(sequences_tensor, 'data/focused/processed_dataset.pt')
print(f"Saved processed dataset to data/focused/processed_dataset.pt")

# Test decoding a few sequences
print("\nSample sequences:")
for i in range(3):
    sequence = sequences[i]
    text = ''.join([idx_to_char[idx] for idx in sequence])
    print(f"Sequence {i}: {text}")