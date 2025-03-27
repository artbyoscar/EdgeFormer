# src.utils.kv_cache_offload

## Functions

### kv_cache_offload

Enable KV cache offloading to disk for a model.

Args:
    model: The EdgeFormer model to enable offloading for
    offload_path: Path to store KV cache files (default: temporary directory)
    kv_cache_dtype: Data type for KV cache (default: same as model)
    
Returns:
    Model with KV cache offloading enabled

```python
kv_cache_offload(model, offload_path=None, kv_cache_dtype=None)
```

### load_kv_cache_from_disk

Load KV cache from disk.

```python
load_kv_cache_from_disk(model, offload_id)
```

### save_kv_cache_to_disk

Save KV cache to disk.

```python
save_kv_cache_to_disk(model, past_key_values, offload_id)
```

