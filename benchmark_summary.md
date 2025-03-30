# EdgeFormer Benchmark Analysis

Analysis generated on: 2025-03-29 19:14:32

Total benchmark results analyzed: 30

## Overall Performance Metrics

### Execution Time
- Minimum: 0.3092
- Maximum: 11.3562
- Mean: 2.7180
- Median: 0.9345
- Standard Deviation: 3.5498

### Tokens Per Second
- Minimum: 291.6578
- Maximum: 1186.9122
- Mean: 747.9750
- Median: 905.0437
- Standard Deviation: 300.2483

### Memory Usage
- Minimum: 307.6367
- Maximum: 883.4086
- Mean: 610.6459
- Median: 585.4121
- Standard Deviation: 230.4466

### Sequence Length
- Minimum: 128.0000
- Maximum: 4096.0000
- Mean: 1561.6000
- Median: 1024.0000
- Standard Deviation: 1445.9683

## Feature Impact Analysis
No feature columns found in the benchmark data.

## Feature Combination Analysis
No feature columns found in the benchmark data.

## Optimal Configurations

### Top Overall Configurations
| Rank | Configuration | Overall Score |
|------|--------------|---------------|
| 1 | sequence_length=512.0 | 0.6796 |
| 2 | sequence_length=1024.0 | 0.5615 |
| 3 | sequence_length=128.0 | 0.2330 |
| 4 | sequence_length=2048.0 | -0.0973 |
| 5 | sequence_length=4096.0 | -1.3767 |

### Recommendations by Use Case

#### For Maximum Speed
- Configuration: sequence_length=1024
- Performance: 1186.9122 tokens/second

#### For Memory Efficiency
- Configuration: sequence_length=128
- Memory Usage: 307.6367 MB

#### For Balanced Performance
- Configuration: sequence_length=512.0
- Overall Score: 0.6796


## Visualizations

Visualizations are available in the `benchmark_results/analysis` directory.

Key visualizations:

- [Performance by Attention Type](benchmark_results/analysis/performance_by_attention_type_20250329-191412.png)
- [Performance vs Sequence Length](benchmark_results/analysis/performance_vs_sequence_length_20250329-191412.png)
- [Feature Impact Dashboard](benchmark_results/analysis/overview_dashboard_20250329-191412.png)
