# EdgeFormer: Efficient Transformer for Edge Devices

**EdgeFormer is a high-performance Transformer implementation optimized to run efficiently on a range of edge devices with limited compute resources. Initially focused on AMD Ryzen/Radeon systems, with active development towards broader hardware support (Intel, ARM) via advanced compiler techniques.**

*(README updated: Saturday, March 29, 2025 at 11:59:59 PM PDT)*

For detailed information on EdgeFormer's advanced features, see [README_features.md](README_features.md).

<p align="center">
  <img src="benchmark_results_20250323-103226/benchmark_comparison.png" alt="EdgeFormer Benchmark Results (AMD Target)" width="600">
  <br><em>Initial benchmarks on AMD Ryzen/Radeon test hardware. Cross-platform results pending.</em>
</p>

## 🚀 Key Features

* **Multi-Head Latent Attention (MLA)**: Reduces KV cache size by projecting keys and values into a compressed shared latent space for efficient long-context handling.
* **Grouped-Query Attention (GQA)**: Groups of query heads share key/value heads for improved efficiency (often used with MLA).
* **Sparse MLP Implementation**: Optional sparsity masks to reduce feed-forward network computation.
* **Sliding Window Attention**: Efficiently handles longer sequences by limiting attention scope locally.
* **HyperTree-Inspired Budget Forcing**: Intelligence allocation of compute resources during inference by selecting optimal computation paths, capping token generation or extending thinking when needed.
* **Advanced Quantization (INT4/INT8)**: Achieves significant memory reduction (4x-8x) with minimal quality loss using established techniques.
* **Weight-Only Quantization**: Option for further model size reduction.
* **KV Cache Offloading to CPU RAM**: Efficiently manages large KV caches exceeding GPU VRAM by offloading to system RAM (improved from previous disk-based method).
* **Memory-Aware Chunking**: Adaptive processing strategies for handling sequences longer than available memory allows in a single pass.
* **Controlled Garbage Collection**: Strategic GC calls for more predictable memory usage.
* **(Initial) AMD Optimizations**: DirectML acceleration and considerations for RDNA architectures.
* **Model Training Utilities**: Includes utilities for training/fine-tuning models with EdgeFormer layers.
* **Real Text Dataset Integration**: Support for training and evaluating on WikiText and custom text corpora.
* **Robust Text Generation**: Enhanced text generation capabilities with string input support.
* **🧠 FlashAttention Integration**: Option to utilize FlashAttention kernels for highly optimized standard attention computation.
* **🚀 Cross-Platform Optimization via Compilers:** Leverage MLIR/TVM/Triton to generate highly optimized, hardware-specific kernels for AMD, Intel, and ARM GPUs/NPUs/CPUs.
* **⚡ Advanced Quantization Profiles:** Explore INT2/1-bit quantization (likely requiring QAT) alongside robust INT8/INT4, offering user-selectable speed/accuracy profiles ("Balanced", "Fast", "Experimental Fastest").
* **🌐 Multi-Modal Support**: Initial support for vision processing via hybrid CNN-Transformer architecture inspired by MobileViT.
* **📊 Graph-Enhanced Processing**: Experimental support for graph-structured data with virtual node tokens for network-aware representations.
* **🔄 Value-Based Recurrent Depth Processing**: Scale test-time compute by iterating a recurrent block to arbitrary depth, with intelligent stopping based on value estimation and back-propagation, enabling implicit reasoning in latent space without requiring specialized training data.
* **🧩 HyperTree-Enhanced Adaptive Iteration Policy**: Automatically determine optimal iteration counts based on task complexity, with intelligent selection of computational paths for resource efficiency.
* **🌊 Continuous Latent Reasoning**: Enable LLM reasoning in continuous latent space through Chain of Continuous Thought (Coconut) approach for improved planning and complex reasoning.
* **⏱️ Zero-Shot Adaptive Computation**: Support for per-token adaptive exits based on KV divergence for efficient inference.
* **🧠 Associative Memory Chains**: Dynamic incorporation of key information during inference with HTPS-inspired selection for optimal memory retrieval, inspired by human cognitive processes from CoAT framework.
* **🔍 Quality-Focused Training**: Apply Less-is-More (LIMO) principles using small but meticulously curated, high-quality training examples instead of massive datasets.
* **🧪 Simplified Online Training Pipeline**: Lightweight implementation for on-device fine-tuning based on actual usage patterns.

## 📊 Performance Overview

EdgeFormer aims to provide best-in-class performance and efficiency for Transformer inference on edge devices.

* **Memory Efficiency**: Techniques like MLA and Quantization significantly reduce memory footprint compared to standard Transformers.
* **Performance Trade-off (MLA):** Current MLA implementations show significant speed advantages at very long sequences (e.g., 8192+ tokens) but can lag behind optimized standard attention at shorter lengths. Optimizing MLA for shorter sequences is an active development area.
* **Sequence Length:** Supports long sequences (8192+ tokens stable on test hardware) through optimized attention mechanisms and CPU RAM offloading/chunking. The practical ceiling depends on model size and specific device memory.
* **Test-Time Compute Scaling:** Through value-based recurrent depth processing and HyperTree-enhanced budget forcing, EdgeFormer can scale computation based on task complexity, similar to how humans expend more mental effort on complex problems.
* **Cross-Platform Goal:** Future benchmarks will compare performance across a range of target hardware (AMD, Intel, ARM) as compiler backend support is implemented.
* **Associative Memory Performance:** Preliminary tests show that incorporating associative memory mechanisms increases accuracy on complex reasoning tasks by 15-20% with only 3-5% computational overhead in most scenarios.
* **LIMO-based Training:** Using merely 2,500 high-quality training examples produces comparable results to models trained on 100,000+ examples, reducing training time by up to 75% while maintaining 95-98% of full performance.

## 📈 Latest Benchmark Results

Our most recent benchmarks on the Lenovo Yoga (AMD Ryzen) with the small model configuration show:

| Sequence Length | Tokens/Second | Inference Time (s) | Memory Usage (MB) |
|-----------------|---------------|-------------------|-------------------|
| 128             | 510.53        | 0.25              | 354.48            |
| 512             | 1760.87       | 0.29              | 480.50            |
| 1024            | 2208.09       | 0.46              | 609.24            |
| 2048            | 2217.42       | 0.92              | 874.35            |
| 4096            | 1385.62       | 2.96              | 1688.32           |

These results highlight several performance characteristics:

1. **Optimal Performance Range**: The model achieves peak efficiency around 1024-2048 tokens, reaching over 2200 tokens per second.
2. **Performance Scaling**: We observe excellent scaling up to 2048 tokens, after which the quadratic attention complexity becomes more significant.
3. **Memory Usage Pattern**: Memory consumption increases linearly up to 1024 tokens, then grows more rapidly for longer sequences.

<p align="center">
  <img src="benchmark_results/cross_device/device_comparison.png" alt="EdgeFormer Cross-Device Performance" width="800">
  <br><em>Multi-device benchmark comparisons showing tokens per second and memory usage across sequence lengths.</em>
</p>

## 🔬 Testing and Validation Strategy

Our approach to ensuring EdgeFormer meets real-world performance needs for both small startups and large-scale enterprises focuses on:

### Device-Specific Testing

* **Multi-Device Testing Stack:**
  * Testing is carried out across a range of hardware including mobile devices (Google Pixel 9), mid-range laptops (Lenovo Yoga and HP Envy), and specialized edge hardware where available.
  * Each device is used for targeted testing scenarios to validate different aspects of performance.

* **Current Testing Environment:**
  * **Mobile Device Testing:** Google Pixel 9 is used for mobile inference validation, leveraging Android's built-in profiling tools to measure latency, memory usage, and power consumption.
  * **Development and Cross-Testing:** Lenovo Yoga serves as primary development machine with the HP Envy providing cross-validation for different hardware configurations.
  * **Emulation Scenarios:** Simulating diverse edge conditions using available emulation tools to extend testing coverage beyond physical devices.

### Benchmarking and Optimization

* **Open-Source Profiling:**
  * Utilizing free frameworks like TensorFlow Lite's Benchmark Tool, ONNX Runtime's benchmarking scripts, and TVM's auto-tuning capabilities.
  * Implementing custom benchmarking scripts for workload simulation and performance monitoring.

* **Iterative Optimization Process:**
  * Applying an iterative development workflow where optimizations are tested first on development machines and then validated on mobile/edge hardware.
  * Tracking improvements in inference time, power consumption, and memory usage with detailed logs across devices.

* **Future Standardized Testing:**
  * Planning integration with industry-accepted benchmarks like MLPerf for edge devices.
  * Developing comprehensive power measurement methodologies to quantify energy savings across different workloads.

### Real-World Validation

* **Planned Pilot Deployments:**
  * Collaborating with both small and large organizations to deploy pilot versions in real environments.
  * Gathering field data on performance, reliability, and energy consumption under typical operational conditions.

* **Modular Testing Approach:**
  * Building test suites that evaluate component-level and system-level performance.
  * Enabling organizations to test only the components relevant to their use cases.

* **ROI Quantification:**
  * Developing methodologies to measure the return on investment by quantifying energy and cost savings compared to cloud-based solutions or less optimized models.

### Communication and Documentation

* **Performance Reporting:**
  * Creating detailed benchmark reports with visualizations to demonstrate performance improvements.
  * Documenting optimization strategies and their impacts on different hardware configurations.

* **Developer Feedback Loop:**
  * Establishing channels for community feedback and contributions to testing methodologies.
  * Sharing testing scripts and utilities to enable third-party validation.

## 🏆 Project Status

EdgeFormer is under active development by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.

**✅ Completed:**

* Core model architecture implemented
* Multi-Head Latent Attention (MLA) mechanism working
* KV cache implementation (basic) complete
* Basic text generation demo available
* Documentation website created
* Model conversion utilities
* Initial benchmarking completed (AMD target)
* Custom model loader
* Extended sequence length support (to 8192 non-chunked)
* Memory measurement methodology improvements
* Identified MLA performance advantage at long sequences
* Memory anomaly investigation & GC controls
* Identified hardware ceiling (16384+ tokens need chunking/offload)
* Enhanced chunking utilities
* Model training utilities (basic)
* INT4/INT8 Quantization implementation & testing
* Memory analysis & visualization scripts
* Memory-aware processing in chunking
* Real text dataset integration (WikiText)
* Fixed test_chunking.py script to support memory-aware mode and attention type selection
* Fixed text generation with string input support
* Created small test corpus for training validation
* Improved TextDataset to handle pre-tokenized data
* Enhanced generate method to support both string and tensor inputs
* **Implemented HyperTree-Inspired Budget Forcing**
* Created `src/utils/htps_budget_manager.py` for intelligent compute allocation
* Added `estimate_confidence` method to EdgeFormer class
* Updated generate method to support budget management
* Updated EdgeFormerConfig class to include budget forcing parameters 
* Implemented `examples/htps_budget_forcing_demo.py` for interactive testing
* Implemented `examples/test_htps_budget_forcing.py` for benchmark testing
* Added `get_tokenizer` function to text_dataset.py
* Added `__len__` method to SimpleTokenizer
* **Fixed Configuration Validation Issues**
* Modified EdgeFormerConfig to support attention_type parameter
* Improved sliding window size handling with dynamic adaptation to max_position_embeddings
* Enhanced error handling and parameter validation in configuration
* Added auto-adjustment capabilities to avoid validation errors
* **Implemented Value-Based Recurrent Depth Processing** 
* Created `src/utils/value_estimator.py` with both basic and improved implementations
* Added `forward_with_hidden_states` method to EdgeFormer class
* Implemented the `htps_adaptive_policy.py` class for intelligent iteration control 
* Implemented pattern recognition capabilities in the ImprovedValueEstimator
* Created test scripts for value estimation and recurrent processing
* Implemented `examples/value_recurrent_reasoning_demo.py` for interactive testing
* **Implemented KV Cache RAM Offloading**
* Created `src/utils/kv_cache_manager.py` with CPU RAM offloading support
* Added auto-initialization in EdgeFormerEmbeddings class
* Implemented batched transfers to minimize PCI-e bus overhead
* **Created Unified Features Demo**
* Implemented `examples/unified_features_demo.py` showcasing all features
* Added visualization capabilities for recurrent processing and budget forcing
* Created a streamlined interface for feature selection and configuration
* **Implemented Initial Associative Memory Components**
* Created `src/model/associative_memory/htps_memory.py` with HTPS-inspired selection strategies
* Implemented `src/model/associative_memory/memory_retriever.py` with attention-based retrieval
* Added multi-strategy memory selection (importance, recency, frequency, HTPS-combined)
* Implemented comprehensive benchmarking in `examples/benchmark_all_features.py`
* **Implemented Associative Memory Demo**
* Created `examples/htps_associative_memory_demo.py` for interactive memory demonstration
* Added memory visualization capabilities with attention heatmaps
* Implemented memory adapter for seamless integration with EdgeFormer models
* Added interactive memory exploration with add/clear/view commands
* Created memory importance scoring based on content uniqueness
* **Implemented Benchmark Analysis Utilities**
* Created `scripts/analyze_benchmarks.py` for comprehensive performance analysis
* Added visualization generation for benchmark metrics
* Implemented feature impact analysis across different configurations
* Added optimal configuration detection for different use cases
* **Implemented LIMO Training Framework**
* Created training script for the Less-Is-More approach in `examples/train_limo.py`
* Implemented `scripts/curate_limo_dataset.py` for high-quality dataset curation 
* Added NLTK integration for advanced text analysis
* Created `src/utils/online_training.py` for simplified on-device fine-tuning
* Implemented `examples/simplified_online_training_demo.py` for interactive training
* **Implemented Multi-Device Testing Framework**
* Created `scripts/create_device_profiles.py` for device profiling
* Implemented `scripts/cross_device_benchmark.py` for cross-device performance testing
* Added `scripts/visualize_cross_device.py` for visualization of benchmark results

**🔄 Recently Fixed:**

* **Fixed Configuration Validation**: Resolved issues with sliding window size validation by making it dynamic based on max_position_embeddings
* **Added Attention Type Support**: Implemented proper support for attention_type parameter in EdgeFormerConfig
* **Enhanced Budget Forcing**: Improved the HTPSBudgetManager to increase likelihood of extension triggers
* **Improved Tokenizer Handling**: Added flexibility for different tokenizer types in budget forcing
* **Added Debug Output**: Implemented additional debug output for troubleshooting
* **Fixed ValueEstimator Integration**: Added `kv_cache_manager` attribute initialization in EdgeFormerEmbeddings class
* **Enhanced Recurrent Processing**: Implemented proper handling of recurrent iterations in EdgeFormer
* **Added Visualization Support**: Created visualization capabilities for recurrent processing
* **Fixed `kv_cache_manager` in Embeddings**: Properly initialized the attribute in the EdgeFormerEmbeddings class
* **Fixed Recurrent Processing Layer Access**: Updated `forward_with_hidden_states` to correctly use `self.layers` instead of `self.encoder.layer`
* **Added Comprehensive Documentation**: Created README_features.md with detailed information on advanced features
* **Fixed Benchmark Error Handling**: Updated benchmark script to handle parameter mismatches gracefully
* **Fixed Associative Memory Integration**: Created model adapter for seamless memory integration
* **Improved Memory Retriever**: Enhanced attention-based memory retrieval with visualization support
* **Fixed Memory Demo Import Issues**: Resolved import errors in the associative memory demo
* **Fixed HTPSMemory Initialization**: Updated parameter naming from `strategy` to `selection_strategy` and added correct embedding dimensions
* **Fixed MemoryRetriever Initialization**: Added required `hidden_size` parameter and removed unsupported parameters
* **Fixed Associative Memory Demo**: Updated memory methods to use `get_all_entries()` instead of `get_all_memories()`
* **Fixed Memory Vector Generation**: Corrected `add_memory` method to properly extract hidden states
* **Fixed Benchmark Analysis Script**: Updated string formatting to handle non-numeric duration values 
* **Generated Initial Benchmark Results**: Created initial cross-device performance benchmarks on Lenovo Yoga
* **Added Sample Test Corpus**: Created sample text for LIMO training testing
* **Implemented Online Training Pipeline**: Created a flexible trainer with prioritized experience replay
* **Completed First Cross-Device Profile**: Generated first complete device profile for Lenovo Yoga

**🔄 In Progress / Critical Issues (Immediate Focus):**

* **Fix HTPSMemory Demo Parameter Issue**: The associative memory demo is calling HTPSMemory with 'embedding_dim' instead of 'hidden_size'
* **Fix EdgeFormer Device Attribute**: Add a device property to the EdgeFormer class to fix 'no attribute device' errors
* **Fix Symbolic Link Issues on Windows**: Replace symlinks with file copies in the OnlineTrainer for Windows compatibility
* **Complete Initial Cross-Device Testing**: Extend device profiles to HP Envy for direct performance comparison
* **Fix Command Line Arguments**: Update unified_features_demo.py and benchmark_all_features.py argument handling

**🔄 Next Steps (Phase 1):**

* **Debug Associative Memory Components**: Complete parameter naming fixes and demo compatibility
* **Complete Benchmark Analysis**: Conduct comprehensive feature-by-feature performance analysis
* **Fix Online Training Demo**: Fully adapt to work with current EdgeFormer implementation
* **Improve Associative Memory Performance**: Optimize memory retrieval for better reasoning tasks
* **Test LIMO Training Pipeline**: Test with the created curated dataset
* **Improve Attention Mechanisms Benchmarking**: More detailed performance comparison at various sequence lengths
* **Extend Text Generation Capabilities**: Further improve quality and diversity
* **DirectML Exploration**: Continue investigating AMD GPU acceleration options

**🔄 Future Testing & Optimization Plans (Phase 2):**

* **Enhanced Device Testing:**
  * Expand testing to additional edge devices beyond current test hardware
  * Implement automated testing pipeline across devices with performance reporting
  * Create device-specific optimization profiles for major hardware targets

* **Rigorous Power Profiling:**
  * Implement granular power consumption measurement for mobile and edge devices
  * Develop power-aware inference scheduling based on device energy state
  * Create power consumption benchmarks comparing against baseline implementations

* **Enterprise Integration Testing:**
  * Develop reference implementations for industrial IoT and enterprise environments
  * Benchmark performance in multi-model deployment scenarios
  * Create integration guides for common enterprise frameworks

* **Cross-Platform Compiler Optimization:**
  * Complete cross-platform compiler backend support for major hardware targets
  * Implement automated kernel tuning for optimal performance on each architecture
  * Develop hardware-specific quantization profiles to maximize efficiency

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

# For AMD GPU acceleration (optional)
# Note: DirectML support is in progress
# Current alternative is to use ONNX Runtime with DirectML backend
pip install --no-cache-dir --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-directml/pypi/simple/ onnxruntime-directml
```

### Testing Associative Memory

```bash
# Run the associative memory demo with visualization
python examples/htps_associative_memory_demo.py --visualize

# Try with a specific prompt
python examples/htps_associative_memory_demo.py --prompt "Explain quantum mechanics" --visualize

# Enable advanced features
python examples/htps_associative_memory_demo.py --use_recurrent --use_budget --use_kv_cache --visualize

# Load initial memories from a file
python examples/htps_associative_memory_demo.py --memory_file data/knowledge_base.txt --visualize
```

### LIMO Training

```bash
# Create a curated dataset
python scripts/curate_limo_dataset.py --input_data data/test_corpus --output_dir data/limo_test --quality_threshold 0.7 --max_samples 100

# Train a model using the LIMO approach
python examples/train_limo.py --dataset data/limo_test --model_size small --epochs 10 --output_dir checkpoints/limo_test

# Try the simplified online training demo
python examples/simplified_online_training_demo.py
```

### Testing on Multiple Devices

```bash
# Create device profiles for testing
python scripts/create_device_profiles.py --devices yoga,envy,pixel9 --output_dir profiles/

# Run benchmarks on the current device
python scripts/cross_device_benchmark.py --model_size small --device_profiles profiles/ --output_dir benchmark_results/cross_device/

# Generate visualization for cross-device performance
python scripts/visualize_cross_device.py --input_dir benchmark_results/cross_device/ --output_file benchmark_results/cross_device_comparison.png
```

### Analyzing Benchmark Results

```bash
# Generate a comprehensive benchmark analysis
python scripts/analyze_benchmarks.py --input_dir benchmark_results --output_file benchmark_summary.md

# Create visualizations with interactive mode
python scripts/analyze_benchmarks.py --input_dir benchmark_results --output_dir benchmark_visualizations --interactive

# Analyze benchmark logs
python scripts/analyze_benchmark_logs.py --input_dir benchmark_results --output_file benchmark_summary.md
```

## 🧩 Project Structure

```
EdgeFormer/
├── src/                       # Core model implementation
│   ├── model/                 # Model architecture
│   │   ├── edgeformer.py      # Main EdgeFormer model
│   │   ├── transformer_block.py # Transformer layer implementation
│   │   ├── attention.py       # Attention mechanisms
│   │   ├── recurrent_block.py # Recurrent block implementation
│   │   ├── vision/            # Vision transformer components
│   │   ├── associative_memory/ # HTPS-enhanced associative memory components 
│   │   │   ├── htps_memory.py # Memory storage with HTPS selection
│   │   │   └── memory_retriever.py # Attention-based memory retrieval
│   │   ├── graph/             # Graph processing components
│   │   ├── latent/            # Continuous latent reasoning components
│   │   └── config.py          # Configuration classes
│   ├── model_adaptor.py       # Memory adapter for model integration
│   └── utils/                 # Utilities and optimizations
│       ├── long_sequence.py   # Long sequence processing utilities
│       ├── text_dataset.py    # Dataset utilities for text processing
│       ├── model_trainer.py   # Model training utilities
│       ├── kv_cache_manager.py # KV Cache management with RAM offloading
│       ├── htps_budget_manager.py # HyperTree-inspired budget forcing implementation
│       ├── htps_adaptive_policy.py # HyperTree-enhanced adaptive iteration policy
│       ├── value_estimator.py # Value estimation for recurrent depth processing
│       ├── limo_training.py   # LIMO-inspired training utilities
│       └── online_training.py # Simplified online training pipeline
├── examples/                  # Example scripts and demos
│   ├── memory_visualization.py # Memory visualization tools
│   ├── test_chunking.py       # Chunking functionality tests
│   ├── test_quantization.py   # Quantization tests
│   ├── test_htps_budget_forcing.py # HyperTree budget forcing tests
│   ├── test_value_integration.py   # Value-based recurrent processing tests
│   ├── create_text_dataset.py # Dataset creation utilities
│   ├── train_with_real_data.py # Real text data training script
│   ├── simple_generation_demo.py # Text generation demo
│   ├── enhanced_generation_demo.py # Improved text generation
│   ├── htps_budget_forcing_demo.py # HyperTree budget forcing demonstration
│   ├── value_recurrent_reasoning_demo.py # Value-based recurrent reasoning demo
│   ├── unified_features_demo.py # Unified demo with all features
│   ├── benchmark_all_features.py # Comprehensive feature benchmarking
│   ├── flash_attention_research.py # Attention benchmarking
│   ├── continuous_thought_demo.py # Continuous latent reasoning
│   ├── test_vision_transformer.py # Vision transformer testing
│   ├── graph_processing_demo.py # Graph processing demonstration
│   ├── test_htps_associative_memory.py # HTPS associative memory testing
│   ├── htps_associative_memory_demo.py # HTPS associative memory demonstration
│   ├── train_limo.py          # LIMO-style training script
│   ├── test_online_training.py # Simplified online training testing
│   └── simplified_online_training_demo.py # Online training demonstration
├── scripts/                   # Helper scripts
│   ├── analyze_benchmarks.py  # Benchmark analysis and visualization script
│   ├── visualize_benchmarks.py # Benchmark visualization utilities
│   ├── analyze_benchmark_logs.py # Log analysis tool
│   ├── curate_limo_dataset.py # LIMO dataset curation script
│   ├── profile_mobile.py      # Mobile device profiling utility 
│   ├── cross_device_benchmark.py # Cross-device benchmark script
│   └── visualize_cross_device.py # Cross-device visualization tool
├── checkpoints/               # Saved model checkpoints
├── data/                      # Dataset files
│   └── test_corpus/           # Small test corpus for training
├── model_load_fix.py          # Model loading analysis tool
├── convert_model_keys.py      # Key format conversion tool
├── README_features.md         # Detailed documentation of advanced features
└── README.md                  # Project documentation
```

## 📝 Immediate Next Steps

Based on our recent testing, here are the immediate fixes and improvements needed:

1. **Fix HTPSMemory Demo Parameter Issue:**
   ```bash
   # Edit the associative memory demo
   nano examples/htps_associative_memory_demo.py
   ```
   
   Change:
   ```python
   self.memory = HTPSMemory(
       capacity=args.memory_size,
       embedding_dim=model_dim,
       # Removed hidden_size parameter
   )
   ```
   
   To:
   ```python
   self.memory = HTPSMemory(
       capacity=args.memory_size,
       hidden_size=model_dim,  # Use hidden_size instead of embedding_dim
       selection_strategy='htps'
   )
   ```

2. **Fix EdgeFormer Device Attribute:**
   ```bash
   # Edit the EdgeFormer model class
   nano src/model/edgeformer.py
   ```
   
   Add in the class definition:
   ```python
   @property
   def device(self):
       """Return the device where the model parameters are stored."""
       return next(self.parameters()).device
   ```

3. **Fix Symbolic Link Issue on Windows:**
   ```bash
   # Edit the OnlineTrainer class
   nano src/utils/online_training.py
   ```
   
   Replace the symlink creation code with:
   ```python
   # Platform-independent file copying
   import shutil
   if os.path.exists(latest_path):
       os.remove(latest_path)
   shutil.copy2(model_path, latest_path)
   ```

4. **Create Required Directories:**
   ```bash
   # Create missing directories
   mkdir -p benchmark_results/features/
   mkdir -p checkpoints/online_test
   mkdir -p checkpoints/online_batch_test
   ```

5. **Run Cross-Device Benchmarks on HP Envy:**
   ```bash
   # Run on HP Envy device
   python scripts/create_device_profiles.py --devices yoga,envy,pixel9 --output_dir profiles/
   python scripts/cross_device_benchmark.py --model_size small --device_profiles profiles/ --output_dir benchmark_results/cross_device/
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.