import torch
import os
import sys

# Print Python and package versions
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Print the content of text_dataset.py
print("Contents of text_dataset.py:")
with open('src/utils/text_dataset.py', 'r') as f:
    content = f.read()
    print(content[:500] + '...' if len(content) > 500 else content)

# Try to load the vocabulary
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

# Try to load the dataset
dataset_path = 'data/focused/text_dataset.pt'
if os.path.exists(dataset_path):
    dataset = torch.load(dataset_path)
    print(f"Dataset loaded, type: {type(dataset)}")
    print(f"Dataset structure: {dataset.keys() if isinstance(dataset, dict) else 'Not a dictionary'}")
else:
    print(f"Dataset file not found at {dataset_path}")

# Try to load the model
model_path = 'checkpoints/final_model.pt'
if os.path.exists(model_path):
    model = torch.load(model_path, map_location='cpu')
    print(f"Model loaded, type: {type(model)}")
    print(f"Model keys: {model.keys() if isinstance(model, dict) else 'Not a dictionary'}")
else:
    print(f"Model file not found at {model_path}")

print("Debug script completed.")
