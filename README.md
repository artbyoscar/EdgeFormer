# EdgeFormer: Efficient Transformer for Edge Devices

**EdgeFormer is a high-performance Transformer implementation optimized to run efficiently on a range of edge devices with limited compute resources. Initially focused on AMD Ryzen/Radeon systems, with active development towards broader hardware support (Intel, ARM) via advanced compiler techniques.**

*(README updated: Thursday, March 27, 2025 at 10:45:23 AM PDT)*

<p align="center">
  <img src="benchmark_results_20250323-103226/benchmark_comparison.png" alt="EdgeFormer Benchmark Results (AMD Target)" width="600">
  <br><em>Initial benchmarks on AMD Ryzen/Radeon test hardware. Cross-platform results pending.</em>
</p>

## 🚀 Key Features

* **Multi-Head Latent Attention (MLA)**: Reduces KV cache size by projecting keys and values into a compressed shared latent space for efficient long-context handling.
* **Grouped-Query Attention (GQA)**: Groups of query heads share key/value heads for improved efficiency (often used with MLA).
* **Sparse MLP Implementation**: Optional sparsity masks to reduce feed-forward network computation.
* **Sliding Window Attention**: Efficiently handles longer sequences by limiting attention scope locally.
* **Advanced Quantization (INT4/INT8)**: Achieves significant memory reduction (4x-8x) with minimal quality loss using established techniques.
* **Weight-Only Quantization**: Option for further model size reduction.
* **KV Cache Offloading to CPU RAM**: Efficiently manages large KV caches exceeding GPU VRAM by offloading to system RAM (improved from previous disk-based method).
* **Memory-Aware Chunking**: Adaptive processing strategies for handling sequences longer than available memory allows in a single pass.
* **Controlled Garbage Collection**: Strategic GC calls for more predictable memory usage.
* **(Initial) AMD Optimizations**: DirectML acceleration and considerations for RDNA architectures.
* **Model Training Utilities**: Includes utilities for training/fine-tuning models with EdgeFormer layers.
* **Real Text Dataset Integration**: Support for training and evaluating on WikiText and custom text corpora.
* **Robust Text Generation**: Enhanced text generation capabilities with string input support.
* **🧠 (Planned) FlashAttention Integration**: Option to utilize FlashAttention kernels for highly optimized standard attention computation.
* **🚀 (Planned) Cross-Platform Optimization via Compilers:** Leverage MLIR/TVM/Triton to generate highly optimized, hardware-specific kernels for AMD, Intel, and ARM GPUs/NPUs/CPUs.
* **⚡ (Planned) Advanced Quantization Profiles:** Explore INT2/1-bit quantization (likely requiring QAT) alongside robust INT8/INT4, offering user-selectable speed/accuracy profiles ("Balanced", "Fast", "Experimental Fastest").

## 📊 Performance Overview

EdgeFormer aims to provide best-in-class performance and efficiency for Transformer inference on edge devices.

* **Memory Efficiency**: Techniques like MLA and Quantization significantly reduce memory footprint compared to standard Transformers.
* **Performance Trade-off (MLA):** Current MLA implementations show significant speed advantages at very long sequences (e.g., 8192+ tokens) but can lag behind optimized standard attention at shorter lengths. Optimizing MLA for shorter sequences is an active development area.
* **Sequence Length:** Supports long sequences (8192+ tokens stable on test hardware) through optimized attention mechanisms and CPU RAM offloading/chunking. The practical ceiling depends on model size and specific device memory.
* **Cross-Platform Goal:** Future benchmarks will compare performance across a range of target hardware (AMD, Intel, ARM) as compiler backend support is implemented.

### Latest Benchmark Results (AMD Target Hardware)

Our latest benchmarks show significant improvements in sequence length handling:

- **Memory Usage**: 
  - Standard Attention: Shows increases from ~64MB at 128 tokens to ~1000MB at 8192 tokens
  - MLA with Sliding Window: Shows increases from ~34MB at 128 tokens to ~1258MB at 8192 tokens
  
- **Inference Time**:
  - At shorter sequences (up to 4096 tokens), Standard Attention maintains faster inference times
  - At 8192 tokens, MLA implementations are significantly faster (14.2-15.7s vs 18.1-20.6s for Standard Attention)
  
- **Memory Anomaly Investigation**:
  - Component-level memory tracking confirmed the memory dip around 4096 tokens is due to garbage collection
  - Standard Attention shows large negative memory values after embeddings (-1004.66 MB at 4096 tokens and -506.97 MB at 4608 tokens)
  - After garbage collection, memory usage stabilizes at lower levels (~200-220 MB for 4096/4608 tokens vs. ~900-1200 MB for 3584 tokens)
  - The anomaly appears to be triggered by PyTorch's internal memory management optimizations at power-of-2 sequence lengths
  
- **Sequence Length Ceiling**:
  - Our testing confirms that 8192 tokens is stable on current hardware
  - Attempts to process 16384 tokens lead to system crashes after approximately 5 minutes
  - This establishes our current practical ceiling for sequence length without chunking

- **Quantization Performance**:
  - INT8 quantization provides a 4x compression ratio (82.52MB → 20.63MB) with negligible speed impact
  - INT4 quantization achieves an 8x compression ratio (82.52MB → 10.31MB) with only a 5% speed penalty
  - Both formats demonstrate excellent speed-memory tradeoffs ideal for edge deployment

- **Chunking Effectiveness**:
  - Successfully processed a 16,384 token sequence in 14.29s using chunking
  - Verified stability across multiple runs with different sequence lengths

- **Training Results**:
  - Successfully trained model on synthetic data (5 epochs)
  - Generated checkpoint files (best_model.pt, final_model.pt) of ~256MB each
  - Implemented real text corpus training with WikiText integration
  - Character-level tokenization support with vocabulary size handling
  - Successfully trained on small test corpus for validating the pipeline

> **Note**: The memory dip at 4096 tokens has been identified as an automatic garbage collection event. We've implemented strategic garbage collection controls to make memory usage more predictable across different sequence lengths.

## 🏆 Project Status

EdgeFormer is under active development.

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

**🔄 In Progress / Near-Term Focus (Phase 1):**

* **Implement KV Cache Offloading to CPU RAM:** Migrating from disk-based approach to RAM-based offloading using the KVCacheManager implementation. (High Priority)
* **Research FlashAttention Integration:** Investigating compatibility with AMD hardware and potential optimization approaches. (High Priority)
* **Improve Attention Mechanisms Benchmarking:** Using the new research script to compare performance across different sequence lengths. (Medium Priority)
* **Extend Text Generation Capabilities:** Further improve text generation quality and diversity with additional sampling strategies. (Medium Priority)
* **DirectML Exploration:** Investigating AMD GPU acceleration options via DirectML or ROCm. (Medium Priority)
* **Enhance Text Dataset & Evaluation:** Improving dataset handling and adding evaluation metrics (Perplexity, etc.).
* **Initial Compiler Backend Research (MLIR/TVM/Triton):** Investigating integration paths.
* **Test Trained Models (Real Data):** Evaluate generation quality post-real data training.
* **Create Initial Technical Report:** Documenting current findings.

**💡 Future Directions (Phases 2-3+):**

* **Full Compiler Integration:** Deep integration with MLIR/TVM/Triton for automated kernel generation.
* **Broad Hardware Support:** Explicit optimization and benchmarking for Intel (GPU/NPU) and ARM (Laptop) platforms.
* **Advanced Quantization Profiles:** Implement and evaluate INT2/1-bit options with QAT support.
* **Mature MoE & Multimodal Support:** Efficiently handle more complex model architectures locally.
* **Enhanced Tooling:** Utilities for easier model optimization, deployment, and profiling.
* **On-Device Personalization/Fine-tuning:** Explore efficient local model adaptation.
* **Mobile/Ultra-Low Power Targets:** Investigate pushing optimizations to more constrained devices.
* **API Stability & Documentation:** Provide robust APIs and comprehensive developer guides.
* **Integration Examples:** Showcasing use with common application frameworks.

## 🔍 Key Findings from Memory Analysis

Our detailed component-level memory tracking has revealed the source of the memory dip at 4096 tokens:

1. **Garbage Collection Triggering**: Large negative memory values (-1004.66 MB at 4096 tokens and -506.97 MB at 4608 tokens) appear immediately after the embedding layer, indicating Python's garbage collector is being triggered.

2. **Memory Recovery Pattern**: After garbage collection, memory usage stabilizes at much lower levels (~200-220 MB for 4096/4608 tokens vs. ~900-1200 MB for 3584 tokens).

3. **Power-of-2 Optimization**: The effect is most pronounced near powers of 2 (4096 = 2^12), suggesting internal memory allocation optimizations in PyTorch or CUDA are triggered at these thresholds.

4. **Sawtooth Memory Pattern**: Our visualizations reveal a sawtooth pattern in memory usage where memory consumption grows steadily until a threshold is reached, triggers garbage collection, and then stabilizes at a lower level.

5. **Sequence Length Ceiling**: Testing confirms that our system cannot reliably process sequences of 16384 tokens (2^14) without crashing, establishing our current practical ceiling without implementing chunked processing.

We've implemented strategic garbage collection calls to give more predictable memory usage patterns across different sequence lengths.

## 🛠️ Recent Improvements

Recent development has focused on several key areas:

* **Fixed Text Generation Pipeline**: Implemented robust text generation with support for both string and tensor inputs in EdgeFormer's generate method
* **Created KV Cache Manager**: Implemented RAM-based key-value cache offloading for efficient long-sequence processing
* **Created FlashAttention Research Script**: Developed a benchmarking tool to compare different attention mechanisms
* **Added Enhanced Generation Demo**: Created a more robust text generation demo with better model loading
* **Fixed TextDataset Handling**: Improved TextDataset to properly handle pre-tokenized data
* **Created Small Test Corpus**: Implemented a quick test corpus approach for faster training validation
* **Added Memory-Aware Processing to Chunking**: Fixed test_chunking.py to better support memory-aware processing with different attention types
* **Enhanced Text Generation**: Improved model loading in text generation scripts
* **Strategic Pivot to Cross-Platform Support**: Planning compiler-based optimization approach
* **Defined Roadmap for Advanced Quantization**: Created structured approach for quantization profiles
* **Refined KV Cache Offloading Strategy**: Moving from disk-based to RAM-based approach

## 🔧 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EdgeFormer.git
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

#### Memory Analysis and Benchmarking

```bash
# Enable detailed memory tracking
export EDGEFORMER_DEBUG=1  # On Windows CMD: set EDGEFORMER_DEBUG=1 or PowerShell: $env:EDGEFORMER_DEBUG=1

# Test memory visualization
python examples/memory_visualization.py --data benchmark_data.json --title "EdgeFormer Memory Analysis" --output-dir plots

# Test chunking functionality with memory-aware processing
python examples/test_chunking.py --sequence_length 16384 --chunk_size 4096 --overlap 512 --memory_aware --attention_type mla

# Test quantization
python examples/test_quantization.py --model_type standard --quantization_type int8
python examples/test_quantization.py --model_type mla --quantization_type int4

# Benchmark attention mechanisms
python examples/flash_attention_research.py --min_seq_length 32 --max_seq_length 2048 --num_lengths 5
```

#### Model Training and Text Generation

```bash
# Create a simple test corpus
python -c "with open('data/small_test.txt', 'w', encoding='utf-8') as f: f.write('EdgeFormer is a custom transformer implementation incorporating Multi-Head Latent Attention optimization to run efficiently on edge devices with limited compute. It\'s specifically designed for AMD Ryzen processors and Radeon graphics. The key features include memory efficiency, performance trade-offs, and maximum sequence length support up to 8192 tokens. This is just a small test file to verify that the training pipeline works correctly.')"

# Create a dataset from the test corpus
python examples/create_text_dataset.py --input_file data/small_test.txt --seq_length 32 --output_dir data --show_samples

# Train on the small test corpus
python examples/train_with_real_data.py --dataset_file data/text_dataset.pt --seq_length 32 --batch_size 2 --epochs 5 --attention_type mla --test_generation --device cpu

# Create a dataset from WikiText (requires datasets library)
python examples/create_text_dataset.py --use_wikitext --seq_length 128 --output_dir data --show_samples

# Train with real text data
python examples/train_with_real_data.py --use_wikitext --seq_length 128 --batch_size 4 --epochs 5 --attention_type mla --test_generation --device cpu

# For faster training on GPU (if available)
python examples/train_with_real_data.py --use_wikitext --seq_length 128 --batch_size 4 --epochs 5 --attention_type mla --test_generation --device cuda

# Generate text with a trained model using the enhanced demo
python examples/enhanced_generation_demo.py --model_path checkpoints/final_model.pt --vocab_path data/vocab.pt --prompt "EdgeFormer is a custom transformer that" --max_length 100 --attention_type mla

# Try different attention mechanisms
python examples/optimized_demo.py --attention_type standard --prompt "EdgeFormer is a custom transformer that" --length 100
python examples/optimized_demo.py --attention_type mla --prompt "EdgeFormer is a custom transformer that" --length 100
python examples/optimized_demo.py --attention_type mla_window --prompt "EdgeFormer is a custom transformer that" --length 100
```

#### Using the GUI Demo

```bash
# Launch the interactive GUI demo
python examples/gui_demo.py
```

#### Model Conversion and Loading

```bash
# Analyze model structure
python model_load_fix.py ./path/to/your/model.pt

# Convert model key format
python convert_model_keys.py --input_path ./path/to/your/model.pt --output_path ./converted_model.pt

# Try loading with custom model loader
python examples/demo_custom_model.py --model_path ./path/to/your/model.pt
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
│   │   └── config.py          # Configuration classes
│   └── utils/                 # Utilities and optimizations
│       ├── long_sequence.py   # Long sequence processing utilities
│       ├── text_dataset.py    # Dataset utilities for text processing
│       ├── model_trainer.py   # Model training utilities
│       └── kv_cache.py        # KV Cache management
├── examples/                  # Example scripts and demos
│   ├── memory_visualization.py # Memory visualization tools
│   ├── test_chunking.py       # Chunking functionality tests
│   ├── test_quantization.py   # Quantization tests
│   ├── create_text_dataset.py # Dataset creation utilities
│   ├── train_with_real_data.py # Real text data training script
│   ├── simple_generation_demo.py # Text generation demo
│   ├── enhanced_generation_demo.py # Improved text generation
│   └── flash_attention_research.py # Attention benchmarking
├── scripts/                   # Helper scripts
├── checkpoints/               # Saved model checkpoints
│   ├── best_model.pt          # Best model based on validation loss
│   ├── final_model.pt         # Final model after training
│   ├── epoch_*.pt             # Models saved at each epoch
│   └── checkpoint_*.pt        # Intermediate checkpoints
├── data/                      # Dataset files
│   ├── text_dataset.pt        # Tokenized dataset
│   └── vocab.pt               # Vocabulary information
├── model_load_fix.py          # Model loading analysis tool
├── convert_model_keys.py      # Key format conversion tool
└── README.md                  # Project documentation
```

## 📝 Next Steps (Immediate Focus)

Based on our recent development and progress, our immediate focus is on the following:

1. **Complete KV Cache Offloading to CPU RAM**
   - Integrate the KVCacheManager into EdgeFormer's forward and generate methods
   - Add configurable memory thresholds for automatic offloading
   - Benchmark performance improvements with various sequence lengths
   - Create a detailed example demonstrating KV cache management

2. **Benchmark Attention Mechanisms Thoroughly**
   - Run comprehensive benchmarks using the flash_attention_research script
   - Identify crossover points where MLA outperforms StandardAttention
   - Create visualizations of performance characteristics
   - Document findings and optimization opportunities

3. **Investigate AMD GPU Acceleration Options**
   - Research compatibility with DirectML or alternative acceleration libraries
   - Explore the possibility of ROCm support for AMD GPUs
   - Create device-specific optimization strategies
   - Document setup procedures for AMD acceleration

4. **Text Generation Quality Improvements**
   - Evaluate text generation quality from trained models
   - Implement additional sampling strategies (nucleus sampling, temperature scheduling)
   - Create interactive demo for comparing generation approaches
   - Document best practices for high-quality text generation

5. **Training Workflow Enhancements**
   - Add progress visualization for training runs
   - Implement early stopping based on validation metrics
   - Add support for learning rate scheduling
   - Create comprehensive training tutorial

6. **Continue FlashAttention Integration Research**
   - Investigate AMD-specific optimizations for attention computation
   - Explore potential alternatives to FlashAttention for non-CUDA hardware
   - Create prototypes of optimized attention implementations
   - Benchmark against current attention mechanisms

These immediate steps will significantly enhance EdgeFormer's performance, usability, and cross-platform capabilities, while preparing for the longer-term roadmap.

## 🚀 Next Steps (Roadmap)

Our broader focus remains on enhancing EdgeFormer's core performance, broadening its applicability across diverse edge hardware, and enabling real-world applications.

**Phase 1: Foundational Enhancements & Core Optimizations**

* **Complete KV Cache Offloading to CPU RAM:** Fully integrate the KVCacheManager implementation with the EdgeFormer model for efficient handling of long sequences exceeding GPU VRAM. (High Priority)
* **Finalize AMD Acceleration Support:** Determine and implement the best approach for AMD GPU acceleration, whether through DirectML, ROCm, or other means. (High Priority)
* **Optimize Attention Mechanisms:** Based on benchmark findings, improve both standard and MLA attention for specific sequence length ranges. (High Priority)
* **Establish Robust Benchmarking Suite:** Complete the standardized benchmarking process across various models, sequence lengths, and quantization levels.
* **Enhance Text Generation Quality:** Add diverse sampling strategies and improve generation quality through additional training.
* **Continue FlashAttention Integration Research:** Find the best approach for optimized attention computation on AMD hardware.

**Phase 2: Cross-Platform Support & Deep Compiler Integration**

* **Deep Compiler Integration (MLIR/TVM/Triton):** Research and integrate a compiler backend to enable automated generation of fused, hardware-specific kernels for maximum performance and easier cross-platform targeting. (High Priority, Foundational)
* **Add Support for Intel Platforms:** Extend optimizations and testing to target Intel integrated GPUs (Arc) and NPUs (Core Ultra) using relevant backends (OpenVINO, DirectML, compiler-generated code). (High Priority for Expansion)
* **Add Support for ARM Platforms:** Begin targeting high-performance ARM laptops (e.g., Snapdragon X Elite / Windows on ARM) using appropriate compilation strategies. (Mid Priority for Expansion)
* **Refine Core Engine APIs:** Ensure EdgeFormer exposes clean, stable, and well-documented APIs for loading models, running inference, and controlling optimization parameters, facilitating integration into external applications. (Supports Ecosystem)

**Phase 3: Advanced Optimizations & Features**

* **Explore & Integrate Advanced Quantization Profiles:** Implement INT2 support and investigate the feasibility of 1-bit/Binary/Ternary networks (potentially requiring QAT). Offer clear profiles (e.g., "Balanced INT8", "Fast INT4", "Experimental INT2") allowing users to trade off accuracy for performance/memory. (High Priority R&D)
* **Optimize MLA for Shorter Sequences:** Revisit MLA performance, applying insights from FlashAttention and compiler optimizations to reduce overhead at sequence lengths < 4096 tokens. (Mid Priority)
* **Support Efficient MoE Inference:** Investigate and implement strategies to efficiently run Mixture-of-Experts models locally via EdgeFormer. (Mid Priority R&D)
* **Explore Multimodal Model Support:** Research adapting EdgeFormer's principles for efficient local inference of text-image models. (Longer Term R&D)

**Phase 4: Ecosystem, Evaluation & Documentation**

* **Build Integration Examples:** Create clear examples demonstrating how to use EdgeFormer within common application frameworks (e.g., Hugging Face `pipeline`, FastAPI server, ONNX Runtime session) - *addresses the "Model Control Program" aspect via integration, not inclusion*. (High Priority for Usability)
* **Develop Showcase Application (e.g., "DevInsight"):** Build a high-value application *using* EdgeFormer to concretely demonstrate its benefits (privacy, performance, offline capability) and drive adoption. (Parallel Product Track)
* **Comprehensive Multi-Hardware Edge Device Testing:** Systematically benchmark performance, stability, and resource usage across the supported range of AMD, Intel, and ARM laptops. (Ongoing)
* **Create Comprehensive Technical Report:** Document architecture, optimization techniques, cross-platform benchmark results, quantization trade-offs, and usage recommendations. (Ongoing Documentation)
* **Enhance Developer Documentation & Community Building:** Improve API references, tutorials, contribution guides, and engage with potential users/contributors.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for discussion. *(Link to CONTRIBUTING.md)*

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgements

This implementation draws inspiration from:

* DeepSeek's Multi-Head Latent Attention paper
* Grouped-Query Attention from Google
* Research into efficient attention mechanisms (e.g., FlashAttention)
* Compiler technologies like Apache TVM, LLVM/MLIR, OpenAI Triton
* Advances in model quantization (including extreme quantization research like BitNet)
* Various transformer optimization techniques for edge devices.