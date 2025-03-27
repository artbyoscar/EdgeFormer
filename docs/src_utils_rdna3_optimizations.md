# src.utils.rdna3_optimizations

## Functions

### create_directml_provider_options_for_rdna3

Create optimized DirectML provider options for RDNA3 GPUs.

Returns:
    A list of provider options for DirectML and CPU providers

```python
create_directml_provider_options_for_rdna3()
```

### create_torch_directml_config_for_rdna3

Create optimized torch-directml configuration for RDNA3 GPUs.

Returns:
    A dictionary with torch-directml configuration options

```python
create_torch_directml_config_for_rdna3()
```

### get_rdna3_device

Get the best device for RDNA3 GPUs.

Returns:
    torch.device: The device to use

```python
get_rdna3_device()
```

### is_rdna3_gpu

Detect if the system has an AMD RDNA3 architecture GPU.

Returns:
    bool: True if RDNA3 GPU is detected, False otherwise

```python
is_rdna3_gpu()
```

### optimize_for_rdna3

Apply RDNA3-specific optimizations to the model.

Args:
    model: The EdgeFormer model to optimize
    
Returns:
    The optimized model

```python
optimize_for_rdna3(model)
```

