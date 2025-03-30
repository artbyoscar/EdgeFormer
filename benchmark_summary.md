# EdgeFormer Benchmark Analysis

Analysis generated on: 2025-03-29 18:38:06

Total benchmark results analyzed: 25

## Overall Performance Metrics

### Execution Time
- Minimum: 0.3092
- Maximum: 11.3562
- Mean: 2.7017
- Median: 0.9219
- Standard Deviation: 3.5692

### Tokens Per Second
- Minimum: 294.6574
- Maximum: 1186.9122
- Mean: 761.9165
- Median: 917.7416
- Standard Deviation: 304.4566

### Memory Usage
- Minimum: 307.6367
- Maximum: 883.4086
- Mean: 610.6400
- Median: 585.4297
- Standard Deviation: 231.3219

### Sequence Length
- Minimum: 128.0000
- Maximum: 4096.0000
- Mean: 1561.6000
- Median: 1024.0000
- Standard Deviation: 1450.9804

## Feature Impact Analysis
No feature columns found in the benchmark data.

## Feature Combination Analysis
No feature columns found in the benchmark data.

## Optimal Configurations

### Top Overall Configurations
| Rank | Configuration | Overall Score |
|------|--------------|---------------|
| 1 | sequence_length=512.0 | 0.6872 |
| 2 | sequence_length=1024.0 | 0.5641 |
| 3 | sequence_length=128.0 | 0.2374 |
| 4 | sequence_length=2048.0 | -0.1080 |
| 5 | sequence_length=4096.0 | -1.3807 |

### Recommendations by Use Case

#### For Maximum Speed
- Configuration: sequence_length=1024
- Performance: 1186.9122 tokens/second

#### For Memory Efficiency
- Configuration: sequence_length=128
- Memory Usage: 307.6367 MB

#### For Balanced Performance
- Configuration: sequence_length=512.0
- Overall Score: 0.6872


## Visualizations

Visualizations are available in the `benchmark_results/analysis` directory.

Key visualizations:

- [Performance by Attention Type](benchmark_results/analysis/performance_by_attention_type_20250329-183714.png)
- [Performance vs Sequence Length](benchmark_results/analysis/performance_vs_sequence_length_20250329-183714.png)
- [Feature Impact Dashboard](benchmark_results/analysis/overview_dashboard_20250329-183714.png)
