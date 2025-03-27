# src.utils.device

## Functions

### export_to_onnx

Export PyTorch model to ONNX format optimized for DirectML.

```python
export_to_onnx(model, onnx_path, input_shape=(1, 512))
```

### get_device

Get the best available device, with special handling for AMD GPUs.

Returns:
    torch.device: The best available device

```python
get_device()
```

### get_optimized_directml_session

Create an optimized ONNX session with DirectML provider.

```python
get_optimized_directml_session(onnx_path)
```

### is_amd_gpu_available

Check if an AMD GPU is available on the system.

```python
is_amd_gpu_available()
```

### optimize_for_amd

Apply optimizations specific to AMD GPUs.

```python
optimize_for_amd(model)
```

### print_device_info

Print detailed information about available compute devices.

```python
print_device_info()
```

### run_with_directml

Run inference using DirectML provider.

```python
run_with_directml(session, input_ids, attention_mask)
```

