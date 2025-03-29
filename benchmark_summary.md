# EdgeFormer Benchmark Analysis

Analysis generated on: 2025-03-29 12:52:41

Total benchmark results analyzed: 20

## Overall Performance Metrics

### Execution Time
- Minimum: 0.2374
- Maximum: 2.9579
- Mean: 0.9770
- Median: 0.4608
- Standard Deviation: 1.0400

### Tokens Per Second
- Minimum: 510.5298
- Maximum: 2240.4862
- Mean: 1607.3451
- Median: 1729.4197
- Standard Deviation: 643.3821

### Memory Usage
- Minimum: 354.3047
- Maximum: 1689.6406
- Mean: 801.6848
- Median: 609.4375
- Standard Deviation: 488.2651

### Sequence Length
- Minimum: 128.0000
- Maximum: 4096.0000
- Mean: 1561.6000
- Median: 1024.0000
- Standard Deviation: 1458.5971

## Feature Impact Analysis
No feature columns found in the benchmark data.

## Feature Combination Analysis
No feature columns found in the benchmark data.

## Optimal Configurations

### Top Overall Configurations
| Rank | Configuration | Overall Score |
|------|--------------|---------------|
| 1 | sequence_length=1024.0 | 0.6159 |
| 2 | sequence_length=512.0 | 0.4861 |
| 3 | sequence_length=2048.0 | 0.2699 |
| 4 | sequence_length=128.0 | -0.0227 |
| 5 | sequence_length=4096.0 | -1.3491 |

### Recommendations by Use Case

#### For Maximum Speed
- Configuration: sequence_length=1024
- Performance: 2240.4862 tokens/second

#### For Memory Efficiency
- Configuration: sequence_length=128
- Memory Usage: 354.3047 MB

#### For Balanced Performance
- Configuration: sequence_length=1024.0
- Overall Score: 0.6159


## Visualizations

Visualizations are available in the `benchmark_visualizations` directory.

Key visualizations:

- [Performance by Attention Type](benchmark_visualizations/performance_by_attention_type_20250329-125141.png)
- [Performance vs Sequence Length](benchmark_visualizations/performance_vs_sequence_length_20250329-125141.png)
- [Feature Impact Dashboard](benchmark_visualizations/overview_dashboard_20250329-125141.png)
