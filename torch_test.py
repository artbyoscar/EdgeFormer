import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")

# Try to create a simple neural network
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU()
)

print("Model created successfully")
print(model)