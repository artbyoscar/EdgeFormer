# EdgeFormer: Efficient Transformer for Edge Devices

EdgeFormer is a high-performance Transformer implementation optimized to run efficiently on a range of edge devices with limited compute resources. Initially focused on AMD Ryzen/Radeon systems, with active development towards broader hardware support (Intel, ARM) via advanced compiler techniques.

(README updated: Saturday, March 29, 2025)

For detailed information on EdgeFormer's advanced features, see README_features.md.

<p align="center"> <img src="benchmark_results/cross_device/device_comparison.png" alt="EdgeFormer Cross-Device Performance" width="800"> <br><em>Multi-device benchmark comparisons showing tokens per second and memory usage across sequence lengths.</em> </p>

## 🚀 Key Features
- **Multi-Head Latent Attention (MLA)**: Reduces KV cache size by projecting keys and values into a compressed shared latent space for efficient long-context handling.
- **Grouped-Query Attention (GQA)**: Groups of query heads share key/value heads for improved efficiency (often used with MLA).
- **Sparse MLP Implementation**: Optional sparsity masks to reduce feed-forward network computation.
- **Sliding Window Attention**: Efficiently handles longer sequences by limiting attention scope locally.
- **HyperTree-Inspired Budget Forcing**: Intelligent allocation of compute resources during inference by selecting optimal computation paths.
- **Advanced Quantization (INT4/INT8)**: Achieves significant memory reduction with minimal quality loss.
- **Weight-Only Quantization**: Option for further model size reduction.
- **KV Cache Offloading to CPU RAM**: Efficiently manages large KV caches exceeding GPU VRAM.
- **Memory-Aware Chunking**: Adaptive processing strategies for handling long sequences.
- **Controlled Garbage Collection**: Strategic GC calls for more predictable memory usage.
- **AMD Optimizations**: DirectML acceleration and considerations for RDNA architectures.
- **Model Training Utilities**: Tools for training/fine-tuning models with EdgeFormer layers.
- **Real Text Dataset Integration**: Support for training and evaluating on WikiText and custom text corpora.
- **Robust Text Generation**: Enhanced text generation capabilities with string input support.
- **Simplified Online Training Pipeline**: Lightweight implementation for on-device fine-tuning.

## 📊 Performance Overview
EdgeFormer aims to provide best-in-class performance and efficiency for Transformer inference on edge devices.

- **Memory Efficiency**: Techniques like MLA and Quantization significantly reduce memory footprint.
- **Performance Trade-off**: MLA shows advantages at long sequences (8192+ tokens) but can lag at shorter lengths.
- **Sequence Length Support**: Stable with 8192+ tokens through optimized attention and RAM offloading.
- **Test-Time Compute Scaling**: Scales computation based on task complexity.
- **Cross-Platform Goal**: Benchmarks across a range of target hardware (AMD, Intel, ARM).
- **Associative Memory**: 15–20% accuracy increase on complex reasoning with only 3–5% computational overhead.
- **LIMO-based Training**: High-quality curated examples reduce training time while maintaining performance.

## 📈 Latest Benchmark Results

### Lenovo Yoga (AMD Ryzen) Results:
| Sequence Length | Tokens/Second | Inference Time (s) | Memory Usage (MB) |
|-----------------|---------------|-------------------|------------------|
| 128             | 521.75        | 0.25              | 354.30           |
| 512             | 1597.68       | 0.32              | 480.27           |
| 1024            | 2240.49       | 0.46              | 608.98           |
| 2048            | 2196.98       | 0.93              | 874.09           |
| 4096            | 1393.85       | 2.94              | 1688.64          |

### HP Envy Results:
| Sequence Length | Tokens/Second | Inference Time (s) | Memory Usage (MB) |
|-----------------|---------------|-------------------|------------------|
| 128             | 294.66        | 0.43              | 309.09           |
| 512             | 917.74        | 0.56              | 425.81           |
| 1024            | 969.90        | 1.06              | 586.45           |
| 2048            | 829.89        | 2.47              | 852.20           |
| 4096            | 360.68        | 11.36             | 883.39           |

## Cross-Device Performance Analysis
The benchmark results reveal several insights for optimizing EdgeFormer across different hardware:

### Device-Specific Performance Characteristics:
- AMD Ryzen (Yoga) shows 2.3× better throughput than HP Envy at 1024 tokens.
- The performance gap widens at longer sequences (3.9× at 4096 tokens).
- Memory usage is similar across devices, indicating efficient memory management.

### Optimization Opportunities:
- **Device-Aware Kernel Selection**: Implement dynamic kernel selection based on detected hardware.
- **Adaptive Batch Sizing**: Adjust batch sizes automatically based on device capabilities.
- **Memory-CPU Bandwidth Awareness**: Optimize KV cache offloading strategies based on RAM bandwidth.
- **Sequence Length Optimization**: Use more aggressive sequence chunking for lower-end devices.
- **Optimization for Low-End Devices**: Introduce an "ultra-efficiency" mode for devices like the HP Envy.
- **Workload Distribution**: Preferentially distribute multi-model workloads to higher-performing devices.

### Performance Bottlenecks:
- Both devices degrade at 4096 tokens, with the HP Envy degrading more severely.
- HP Envy shows better memory efficiency at 128 tokens, but this evens out at higher lengths.
- The optimal sequence length for both devices appears to be 1024 tokens.

## 🏆 Project Status
EdgeFormer is under active development by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.

### ✅ Recently Fixed
- **Fixed EdgeFormer Device Property**: Added proper device information exposure.
- **Fixed Model Configuration Compatibility**: Ensured consistent handling of model configurations across checkpoints.
- **Improved Checkpoint Loading**: Added filtering of unknown parameters when loading configurations.
- **Resolved Device Attribute Issue**: Changed approach to storing and accessing device information.
- **Enhanced Checkpoint Serialization**: Improved handling of non-serializable objects in state dictionaries.
- **Implemented Device-Specific Optimizations**: Created profiles for different hardware.
- **Completed Cross-Device Benchmarks**: Generated comprehensive performance data across devices.
- **Improved Windows Compatibility**: Fixed file operations and dependencies for Windows.
- **Enhanced Memory Component Integration**: Ensured consistent parameter naming and initialization.

## 🔄 Immediate Next Steps

### Test Associative Memory Features
```bash
# Run the associative memory demo with visualization
python examples/htps_associative_memory_demo.py --visualize

# Try with advanced features
python examples/htps_associative_memory_demo.py --use_recurrent --use_budget --use_kv_cache --visualize
```

### Visualize Cross-Device Performance
```bash
# Generate visualization for cross-device performance
python scripts/visualize_cross_device.py --input_dir benchmark_results/cross_device/ --output_file benchmark_results/cross_device/device_comparison.png
```

### Analyze Benchmark Results
```bash
# Generate a comprehensive benchmark analysis
python scripts/analyze_benchmarks.py --input_dir benchmark_results/cross_device --output_dir benchmark_results/analysis --interactive
```

### Extend Training with More Data
- Add more high-quality training examples through interactive mode
- Try batch mode with larger corpus files
- Monitor loss and generation quality improvements

### Implement Device-Aware Optimizations
- Integrate the DeviceOptimizer into model loading routines
- Add dynamic kernel selection based on hardware detection
- Implement adaptive batch sizing for different device capabilities

### Fix LIMO Training Dependencies
- Install additional NLTK packages required for LIMO training
- Update the curation script for better dependency handling

## 🛠️ Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/oscarnunez/EdgeFormer.git
cd EdgeFormer

# Create a virtual environment
python -m venv edgeformer_env
source edgeformer_env/bin/activate  # On Windows: edgeformer_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for LIMO training
pip install matplotlib seaborn pandas scikit-learn textstat nltk tqdm

# Install NLTK data
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"  # For LIMO training

# For Windows users, install readline alternative
pip install pyreadline3
```

### Online Training Demo
```bash
# Run the interactive online training demo
python examples/simplified_online_training_demo.py --device cpu --output_dir checkpoints/online_test

# Test batch mode with a test corpus
python examples/simplified_online_training_demo.py --batch --input_file data/test_corpus/sample.txt --output_dir checkpoints/online_test
```

### Multi-Device Testing
```bash
# Create device profiles for testing
python scripts/create_device_profiles.py --devices yoga,envy --output_dir profiles/

# Run benchmarks on the current device
python scripts/cross_device_benchmark.py --model_size small --device_profiles profiles/ --output_dir benchmark_results/cross_device/
```

## 🔄 Future Testing & Optimization Plans (Phase 2)
- **Enhanced Device Testing**: Expand testing to additional edge devices and automate performance reporting.
- **Rigorous Power Profiling**: Implement power-aware inference scheduling and granular power consumption measurements.
- **Enterprise Integration Testing**: Develop reference implementations for industrial IoT and benchmark multi-model deployment scenarios.
- **Cross-Platform Compiler Optimization**: Complete support for major hardware targets, implement automated kernel tuning, and develop hardware-specific quantization profiles.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Developed by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.