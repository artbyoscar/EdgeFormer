"""
EdgeFormer Compression Package
Provides INT4 quantization and model compression utilities
"""

from .int4_quantization import (
    INT4Quantizer,
    EdgeFormerINT4Linear,
    compress_model_int4,
    test_quantization_accuracy
)

from .utils import (
    calculate_model_size,
    count_parameters,
    analyze_layer_sizes,
    get_compression_candidates,
    estimate_compression_benefit,
    benchmark_inference_speed,
    compare_model_outputs,
    save_compression_report,
    load_compression_report,
    print_compression_summary,
    create_compression_config,
    get_memory_usage,
    reset_memory_stats
)

__all__ = [
    # INT4 Quantization
    'INT4Quantizer',
    'EdgeFormerINT4Linear', 
    'compress_model_int4',
    'test_quantization_accuracy',
    
    # Utilities
    'calculate_model_size',
    'count_parameters',
    'analyze_layer_sizes',
    'get_compression_candidates',
    'estimate_compression_benefit',
    'benchmark_inference_speed',
    'compare_model_outputs',
    'save_compression_report',
    'load_compression_report',
    'print_compression_summary',
    'create_compression_config',
    'get_memory_usage',
    'reset_memory_stats'
]
