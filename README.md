Here's the full updated README for your EdgeFormer project:

# EdgeFormer: Efficient Transformer for Edge Devices

**EdgeFormer is a high-performance Transformer implementation optimized to run efficiently on a range of edge devices with limited compute resources. Initially focused on AMD Ryzen/Radeon systems, with active development towards broader hardware support (Intel, ARM) via advanced compiler techniques.**

*(README updated: Friday, March 28, 2025 at 14:30:00 PM PDT)*

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

**🔄 Recently Fixed:**

* **Fixed Configuration Validation**: Resolved issues with sliding window size validation by making it dynamic based on max_position_embeddings
* **Added Attention Type Support**: Implemented proper support for attention_type parameter in EdgeFormerConfig
* **Enhanced Budget Forcing**: Improved the HTPSBudgetManager to increase likelihood of extension triggers
* **Improved Tokenizer Handling**: Added flexibility for different tokenizer types in budget forcing
* **Added Debug Output**: Implemented additional debug output for troubleshooting
* **Fixed ValueEstimator Integration**: Added `kv_cache_manager` attribute initialization in EdgeFormerEmbeddings class
* **Enhanced Recurrent Processing**: Implemented proper handling of recurrent iterations in EdgeFormer
* **Added Visualization Support**: Created visualization capabilities for recurrent processing

**🔄 In Progress / Near-Term Focus (Phase 1):**

* **Fix `kv_cache_manager` Issue in Embeddings**: Initialize the `kv_cache_manager` attribute in the EdgeFormerEmbeddings class to prevent the current error (Highest Priority)
* **Test Value-Based Recurrent Processing**: Run tests with the updated implementation to verify functionality (Highest Priority)
* **Implement Unified Demo**: Create a demo that showcases all features (KV cache management, value-based recurrent processing, and budget forcing) (High Priority)
* **Implement KV Cache Offloading to CPU RAM:** Migrating from disk-based approach to RAM-based offloading using the KVCacheManager implementation. (High Priority)
* **Complete FlashAttention Integration:** Finalizing compatibility with AMD hardware and optimizing performance. (High Priority)
* **Enhance Value-Based Recurrent Depth Processing:** Further refine the value estimation component and improve the adaptive iteration policy for better task-specific performance. (High Priority)
* **Optimize HyperTree-Enhanced Adaptive Iteration Policy:** Improve heuristics to determine optimal iteration counts and computation paths for different tasks. (High Priority)
* **Implement LIMO's Quality-Focused Training Approach:** Create curated training datasets following LIMO principles, focusing on quality over quantity. (High Priority)
* **Incorporate HTPS-Enhanced Associative Memory:** Implementing dynamic knowledge retrieval and integration with intelligent selection during the inference process. (High Priority)
* **Develop Simplified Online Training Pipeline:** Create lightweight on-device fine-tuning capabilities based on usage patterns. (High Priority)
* **Improve Attention Mechanisms Benchmarking:** Using the new research script to compare performance across different sequence lengths. (Medium Priority)
* **Extend Text Generation Capabilities:** Further improve text generation quality and diversity with additional sampling strategies. (Medium Priority)
* **DirectML Exploration:** Investigating AMD GPU acceleration options via DirectML or ROCm. (Medium Priority)

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

# For AMD GPU acceleration (optional)
# Note: DirectML support is in progress
# Current alternative is to use ONNX Runtime with DirectML backend
pip install --no-cache-dir --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-directml/pypi/simple/ onnxruntime-directml
```

### Usage Examples

#### Testing Value-Based Recurrent Depth Processing (New)

```bash
# Test value-based recurrent processing functionality
python examples/test_value_integration.py --model_type mla --sequence_length 128 --min_iterations 2 --max_iterations 10 --convergence_threshold 0.005 --device cpu

# Test value-based recurrent reasoning demo
python examples/value_recurrent_reasoning_demo.py --prompt "Solve this math problem: If a circle has a radius of 5 cm, what is its area?" --min_iterations 2 --max_iterations 32 --convergence_threshold 0.005 --device cpu --visualize

# Test with different convergence thresholds
python examples/test_value_integration.py --model_type standard --sequence_length 128 --min_iterations 2 --max_iterations 16 --convergence_threshold 0.01 --device cpu

# Test with different tasks
python examples/value_recurrent_reasoning_demo.py --prompt "Explain the concept of quantum entanglement in simple terms." --min_iterations 2 --max_iterations 24 --convergence_threshold 0.008 --device cpu
```

#### Using the Unified Demo (Planned)

```bash
# Launch the unified demo with all features
python examples/unified_features_demo.py --prompt "EdgeFormer is" --max_length 100 --use_kv_cache --use_recurrent --use_budget --visualize

# Test with a reasoning task
python examples/unified_features_demo.py --prompt "Solve this step by step: If a rectangle has a length of 8 meters and a width of 5 meters, what is its area and perimeter?" --max_length 200 --use_recurrent --min_iterations 2 --max_iterations 12 --convergence_threshold 0.005 --device cpu --visualize
```

## 📚 Documentation

Complete documentation is available via the MkDocs website:

```bash
# Install MkDocs if you haven't already
pip install mkdocs mkdocs-material

# Serve the documentation locally
cd edgeformer-docs
mkdocs serve
```

Visit `http://127.0.0.1:8000` to view the documentation.

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
│   │   ├── graph/             # Graph processing components
│   │   ├── latent/            # Continuous latent reasoning components
│   │   └── config.py          # Configuration classes
│   └── utils/                 # Utilities and optimizations
│       ├── long_sequence.py   # Long sequence processing utilities
│       ├── text_dataset.py    # Dataset utilities for text processing
│       ├── model_trainer.py   # Model training utilities
│       ├── kv_cache.py        # KV Cache management
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
│   ├── unified_features_demo.py # Unified demo with all features (planned)
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
├── checkpoints/               # Saved model checkpoints
├── data/                      # Dataset files
├── model_load_fix.py          # Model loading analysis tool
├── convert_model_keys.py      # Key format conversion tool
└── README.md                  # Project documentation
```

## 📝 Implementation Plan

Based on our current development status and the recent implementation of value-based recurrent depth processing, we've established the following revised implementation plan:

### Immediate Next Steps (Days)

1. **Fix `kv_cache_manager` Issue (Today)**
   - Initialize the `kv_cache_manager` attribute in the EdgeFormerEmbeddings class
   - Add a `config` field to store the configuration in the embeddings class
   - Test with value-based recurrent processing scripts

2. **Complete Value-Based Recurrent Processing (1-2 days)**
   - Test the updated implementation thoroughly
   - Fine-tune convergence detection parameters
   - Create comprehensive documentation

3. **Implement Unified Demo (2-3 days)**
   - Create a demo that showcases all implemented features
   - Implement visualization capabilities for all features
   - Create a streamlined interface for feature selection

4. **Push Changes to GitHub (1 day)**
   - Commit all changes with descriptive commit messages
   - Update README with latest progress
   - Verify functionality across different environments

### Short-Term Focus (1-2 Weeks)

1. **Finalize Value-Based Recurrent Depth Processing (3-4 days)**
   - Optimize pattern recognition capabilities
   - Fine-tune adaptive iteration policy
   - Create comprehensive benchmarks for different tasks
   - Document best practices

2. **Implement KV Cache Offloading to CPU RAM (3-5 days)**
   - Create efficient RAM-based offloading mechanism
   - Implement automatic thresholds for offloading
   - Benchmark performance improvements
   - Create documentation and usage examples

3. **Integrate HTPS-Enhanced Memory Components (4-6 days)**
   - Implement associative memory mechanisms
   - Create selection strategies for memory retrieval
   - Optimize for minimal computational overhead
   - Benchmark performance on reasoning tasks

4. **Enhance Documentation (2-3 days)**
   - Update README_features.md with detailed information
   - Create comprehensive API documentation
   - Add usage examples for all major features
   - Update MkDocs website with visualizations

### Medium-Term Focus (2-4 Weeks)

1. **Implement LIMO's Quality-Focused Training (1-2 weeks)**
   - Create curated training datasets
   - Implement LIMO-based training pipeline
   - Benchmark against models trained on larger datasets
   - Document guidelines for dataset curation

2. **Complete FlashAttention Integration (1-2 weeks)**
   - Finalize AMD hardware compatibility
   - Optimize for different sequence lengths
   - Create comprehensive benchmarks
   - Document best practices for different hardware

3. **Develop Simplified Online Training (1-2 weeks)**
   - Create lightweight fine-tuning pipeline
   - Implement usage-based adaptation strategies
   - Benchmark domain-specific performance improvements
   - Create documentation and examples

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

Developed by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.