
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
    
    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        return 1

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
