import torch
print(f'PyTorch version: {torch.__version__}')
print(f'Device: {torch.device(\
cuda\ if torch.cuda.is_available() else \cpu\)}')
x = torch.rand(2, 3)
print(f'Random tensor:\\n{x}')

