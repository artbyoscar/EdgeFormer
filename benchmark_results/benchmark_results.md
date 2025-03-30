# EdgeFormer Quantization Benchmark Results

## Memory Usage (MB)

| Model Size | FP32 | INT8 | INT4 | INT8 Compression | INT4 Compression |
|------------|------|------|------|------------------|------------------|
| 32 | 6.59 | 6.40 | N/A | 1.03x | N/A |
| 64 | 13.55 | 12.79 | N/A | 1.06x | N/A |
| 128 | 28.58 | 25.56 | N/A | 1.12x | N/A |


## Inference Time (ms)

| Model Size | FP32 | INT8 | INT4 | INT8 Speed Ratio | INT4 Speed Ratio |
|------------|------|------|------|------------------|------------------|
| 32 | 3.08 | 6.59 | N/A | 0.47x | N/A |
| 64 | 3.00 | 5.00 | N/A | 0.60x | N/A |
| 128 | 3.50 | 6.52 | N/A | 0.54x | N/A |
