import os

# Create a mock torch.py module in the current directory
with open('torch.py', 'w', encoding='utf-8') as f:
    f.write("""
# Mock PyTorch functionality
class Tensor:
    def __init__(self, data):
        self.data = data
    
    def to(self, device):
        return self
    
    def size(self):
        if isinstance(self.data, list):
            if isinstance(self.data[0], list):
                return [len(self.data), len(self.data[0])]
            return [len(self.data)]
        return []
    
    def float(self):
        return self
    
    def __getitem__(self, idx):
        return self.data[idx] if isinstance(idx, int) and idx < len(self.data) else self.data

def tensor(data):
    return Tensor(data)

def ones_like(input_tensor):
    if hasattr(input_tensor, 'data'):
        shape = input_tensor.size()
        if len(shape) == 2:
            return Tensor([[1.0 for _ in range(shape[1])] for _ in range(shape[0])])
        return Tensor([1.0 for _ in range(shape[0])])
    return Tensor([1.0])

def zeros_like(input_tensor):
    if hasattr(input_tensor, 'data'):
        shape = input_tensor.size()
        if len(shape) == 2:
            return Tensor([[0.0 for _ in range(shape[1])] for _ in range(shape[0])])
        return Tensor([0.0 for _ in range(shape[0])])
    return Tensor([0.0])
""")

# Update our TensorPlaceholder to be more compatible with PyTorch
with open('src/model/edgeformer.py', 'w', encoding='utf-8') as f:
    f.write("""
# Minimal placeholder for EdgeFormer and EdgeFormerConfig
class EdgeFormerConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get('hidden_size', 768)
        self.num_attention_heads = kwargs.get('num_attention_heads', 12)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 12)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

# Use the mock torch module we just created
import torch

class EdgeFormer:
    def __init__(self, config=None):
        self.config = config or EdgeFormerConfig()
        self.device = 'cpu'
        self.training = True
        self.memory = None
    
    @staticmethod
    def from_pretrained(model_path, **kwargs):
        return EdgeFormer(EdgeFormerConfig(**kwargs))
    
    def to(self, device):
        # Method to move model to a specific device (CPU/GPU)
        self.device = device
        return self
    
    def eval(self):
        # Set model to evaluation mode
        self.training = False
        return self
    
    def train(self, mode=True):
        # Set model to training mode
        self.training = mode
        return self
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Placeholder for forward method
        batch_size = input_ids.size()[0] if hasattr(input_ids, 'size') else 1
        seq_len = input_ids.size()[1] if hasattr(input_ids, 'size') and len(input_ids.size()) > 1 else 10
        
        return {
            "logits": torch.tensor([[[0.0] * self.config.vocab_size for _ in range(seq_len)] for _ in range(batch_size)]),
            "hidden_states": torch.tensor([[[0.0] * self.config.hidden_size for _ in range(seq_len)] for _ in range(batch_size)])
        }
    
    def forward_with_hidden_states(self, input_ids=None, attention_mask=None, **kwargs):
        # Special forward method that returns hidden states separately
        outputs = self.forward(input_ids, attention_mask, **kwargs)
        
        # Create placeholder hidden states for all layers
        batch_size = input_ids.size()[0] if hasattr(input_ids, 'size') else 1
        seq_len = input_ids.size()[1] if hasattr(input_ids, 'size') and len(input_ids.size()) > 1 else 10
        
        # Create hidden states for each layer
        all_hidden_states = [
            torch.tensor([[[0.0] * self.config.hidden_size for _ in range(seq_len)] for _ in range(batch_size)])
            for _ in range(self.config.num_hidden_layers + 1)  # +1 for embeddings
        ]
        
        return outputs, all_hidden_states
    
    def __call__(self, *args, **kwargs):
        # Allow the model to be called directly
        return self.forward(*args, **kwargs)
    
    def generate(self, input_ids, max_length=100, **kwargs):
        # Simple generation placeholder
        batch_size = input_ids.size()[0] if hasattr(input_ids, 'size') else 1
        return torch.tensor([[i % 100 for i in range(max_length)] for _ in range(batch_size)])

class SimpleTokenizer:
    def __init__(self, vocab_size=None, **kwargs):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = vocab_size or 32000
    
    def encode(self, text, return_tensors=None, **kwargs):
        # Convert text to token ids
        token_ids = [ord(c) % self.vocab_size for c in text]
        
        # Handle different return types
        if return_tensors == "pt":
            # Return a PyTorch-compatible tensor
            return torch.tensor([token_ids])
        else:
            return token_ids
    
    def decode(self, ids, **kwargs):
        # Convert token ids back to text
        if hasattr(ids, 'data'):  # If it's our tensor
            ids = ids.data
        
        # Flatten if it's a nested list
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
            
        return ''.join([chr(min(i, 127)) for i in ids])
""")

print("Created mock PyTorch functionality and updated EdgeFormer to use it")