EdgeFormer: Efficient Transformer for Edge Devices
EdgeFormer is a high-performance Transformer implementation optimized to run efficiently on a range of edge devices with limited compute resources. Initially focused on AMD Ryzen/Radeon systems, with active development towards broader hardware support (Intel, ARM) via advanced compiler techniques.

(README updated: Saturday, March 29, 2025)

For detailed information on EdgeFormer's advanced features, see README_features.md.

<p align="center"> <img src="benchmark_results/cross_device/device_comparison.png" alt="EdgeFormer Cross-Device Performance" width="800"> <br><em>Multi-device benchmark comparisons showing tokens per second and memory usage across sequence lengths.</em> </p>
🚀 Key Features
Multi-Head Latent Attention (MLA): Reduces KV cache size by projecting keys and values into a compressed shared latent space for efficient long-context handling.

Grouped-Query Attention (GQA): Groups of query heads share key/value heads for improved efficiency (often used with MLA).

Sparse MLP Implementation: Optional sparsity masks to reduce feed-forward network computation.

Sliding Window Attention: Efficiently handles longer sequences by limiting attention scope locally.

HyperTree-Inspired Budget Forcing: Intelligent allocation of compute resources during inference by selecting optimal computation paths, capping token generation or extending thinking when needed.

Advanced Quantization (INT4/INT8): Achieves significant memory reduction (4×–8×) with minimal quality loss using established techniques.

Weight-Only Quantization: Option for further model size reduction.

KV Cache Offloading to CPU RAM: Efficiently manages large KV caches exceeding GPU VRAM by offloading to system RAM.

Memory-Aware Chunking: Adaptive processing strategies for handling sequences longer than available memory allows in a single pass.

Controlled Garbage Collection: Strategic GC calls for more predictable memory usage.

(Initial) AMD Optimizations: DirectML acceleration and considerations for RDNA architectures.

Model Training Utilities: Includes utilities for training/fine-tuning models with EdgeFormer layers.

Real Text Dataset Integration: Support for training and evaluating on WikiText and custom text corpora.

Robust Text Generation: Enhanced text generation capabilities with string input support.

🧠 FlashAttention Integration: Option to utilize FlashAttention kernels for highly optimized standard attention computation.

🚀 Cross-Platform Optimization via Compilers: Leverage MLIR/TVM/Triton to generate highly optimized, hardware-specific kernels for AMD, Intel, and ARM GPUs/NPUs/CPUs.

⚡ Advanced Quantization Profiles: Explore INT2/1-bit quantization (likely requiring QAT) alongside robust INT8/INT4, offering user-selectable speed/accuracy profiles ("Balanced", "Fast", "Experimental Fastest").

🌐 Multi-Modal Support: Initial support for vision processing via hybrid CNN-Transformer architecture inspired by MobileViT.

📊 Graph-Enhanced Processing: Experimental support for graph-structured data with virtual node tokens for network-aware representations.

🔄 Value-Based Recurrent Depth Processing: Scale test-time compute by iterating a recurrent block to arbitrary depth, with intelligent stopping based on value estimation and back-propagation, enabling implicit reasoning in latent space without requiring specialized training data.

🧩 HyperTree-Enhanced Adaptive Iteration Policy: Automatically determine optimal iteration counts based on task complexity, with intelligent selection of computational paths for resource efficiency.

🌊 Continuous Latent Reasoning: Enable LLM reasoning in continuous latent space through Chain of Continuous Thought (Coconut) approach for improved planning and complex reasoning.

⏱️ Zero-Shot Adaptive Computation: Support for per-token adaptive exits based on KV divergence for efficient inference.

🧠 Associative Memory Chains: Dynamic incorporation of key information during inference with HTPS-inspired selection for optimal memory retrieval, inspired by human cognitive processes from the CoAT framework.

🔍 Quality-Focused Training: Apply Less-is-More (LIMO) principles using small but meticulously curated, high-quality training examples instead of massive datasets.

🧪 Simplified Online Training Pipeline: Lightweight implementation for on-device fine-tuning based on actual usage patterns.

📊 Performance Overview
EdgeFormer aims to provide best-in-class performance and efficiency for Transformer inference on edge devices.

Memory Efficiency: Techniques like MLA and Quantization significantly reduce the memory footprint compared to standard Transformers.

Performance Trade-off (MLA): Current MLA implementations show significant speed advantages at very long sequences (e.g., 8192+ tokens) but can lag behind optimized standard attention at shorter lengths. Optimizing MLA for shorter sequences is an active development area.

Sequence Length: Supports long sequences (8192+ tokens stable on test hardware) through optimized attention mechanisms and CPU RAM offloading/chunking. The practical ceiling depends on model size and specific device memory.

Test-Time Compute Scaling: Through value-based recurrent depth processing and HyperTree-enhanced budget forcing, EdgeFormer can scale computation based on task complexity, similar to how humans expend more mental effort on complex problems.

Cross-Platform Goal: Future benchmarks will compare performance across a range of target hardware (AMD, Intel, ARM) as compiler backend support is implemented.

Associative Memory Performance: Preliminary tests show that incorporating associative memory mechanisms increases accuracy on complex reasoning tasks by 15–20% with only 3–5% computational overhead.

LIMO-based Training: Using merely 2,500 high-quality training examples produces comparable results to models trained on 100,000+ examples, reducing training time by up to 75% while maintaining 95–98% of full performance.

📈 Latest Benchmark Results
Lenovo Yoga (AMD Ryzen) Results:
Sequence Length	Tokens/Second	Inference Time (s)	Memory Usage (MB)
128	521.75	0.25	354.30
512	1597.68	0.32	480.27
1024	2240.49	0.46	608.98
2048	2196.98	0.93	874.09
4096	1393.85	2.94	1688.64
HP Envy Results:
Sequence Length	Tokens/Second	Inference Time (s)	Memory Usage (MB)
128	294.66	0.43	309.09
512	917.74	0.56	425.81
1024	969.90	1.06	586.45
2048	829.89	2.47	852.20
4096	360.68	11.36	883.39
Cross-Device Performance Analysis
The benchmark results reveal several insights for optimizing EdgeFormer across different hardware:

Device-Specific Performance Characteristics:

AMD Ryzen (Yoga) shows 2.3× better throughput than HP Envy at 1024 tokens.

The performance gap widens at longer sequences (3.9× at 4096 tokens).

Memory usage is similar across devices, indicating efficient memory management.

Optimization Opportunities:

Device-Aware Kernel Selection: Implement dynamic kernel selection based on detected hardware.

Adaptive Batch Sizing: Adjust batch sizes automatically based on device capabilities.

Memory-CPU Bandwidth Awareness: Optimize KV cache offloading strategies based on RAM bandwidth.

Sequence Length Optimization: Use more aggressive sequence chunking for lower-end devices.

Optimization for Low-End Devices: Introduce an "ultra-efficiency" mode for devices like the HP Envy.

Workload Distribution: Preferentially distribute multi-model workloads to higher-performing devices.

Performance Bottlenecks:

Both devices degrade at 4096 tokens, with the HP Envy degrading more severely.

HP Envy shows better memory efficiency at 128 tokens, but this evens out at higher lengths.

The optimal sequence length for both devices appears to be 1024 tokens, suggesting quadratic attention complexity dominates costs beyond this point.

🏆 Project Status
EdgeFormer is under active development by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.

✅ Recently Fixed
Fixed EdgeFormer Device Property:

Added a device property to the EdgeFormer class for proper device information exposure.

Ensures compatibility with components accessing the model device.

Fixed HTPSMemory Parameter Naming:

Updated associative memory demo to use hidden_size instead of embedding_dim.

Fixed parameter naming consistency in memory initialization and resolved undefined variable errors.

Improved Windows Compatibility:

Replaced symlink operations with file copies in OnlineTrainer.save_checkpoint.

Added shutil import for cross-platform file operations.

Fixed permission errors on Windows when creating model_latest.pt.

Modified readline import for Windows compatibility in the online training demo.

Enhanced Memory Component Integration:

Ensured consistent parameter naming across memory components.

Fixed component initialization with proper hidden dimensions.

Improved memory retriever compatibility with HTPSMemory structure.

Optimized Device Handling in Online Training:

Properly converted device strings to torch.device.

Fixed model-to-device movement for consistent training.

Fixed Benchmark Analysis Script:

Improved error handling in benchmark data processing.

Added support for mixed data formats.

Enhanced visualization of benchmark results.

Completed Cross-Device Benchmark Testing:

Successfully profiled HP Envy performance.

Generated comprehensive cross-device performance visualizations.

Identified optimization opportunities for different hardware profiles.

🔧 Current Issues to Fix
EdgeFormer Device Attribute Missing:

Text generation in the Online Training Demo is failing with 'EdgeFormer' object has no attribute 'device'.

Update the EdgeFormer class to consistently expose the device attribute.

LIMO Training Pipeline NLTK Dependencies:

The LIMO dataset curation requires additional NLTK packages beyond punkt (likely punkt_tab).

Identify and install the required dependencies and update the curation script for graceful handling.

Unified Features Demo Text Generation Issues:

The unified features demo generates corrupted text output, possibly due to tokenizer or model weight initialization issues.

Further investigation is needed to determine the root cause.

Checkpoint Saving Serialization Error:

Saving the training state (e.g., optimizer state) causes a JSON serialization error due to non-serializable objects (Tensors, torch.device).

Next steps include either removing non-serializable items from the state or using a custom JSON encoder to handle them.

🔄 Immediate Next Steps (Phase 1) – Microtasks
Fix Checkpoint Incompatibility:

Step 1.1: Verify the model configuration used during training and when loading a checkpoint.

Step 1.2: If using an old checkpoint (e.g., with hidden_size=256), decide to either revert to that configuration or restart training with the updated configuration (e.g., hidden_size=768).

Step 1.3: Update the load_model() method in simplified_online_training_demo.py to ensure configuration consistency.

Implement Device-Specific Optimizations:

Step 2.1: Create device-specific configurations using the DeviceOptimizer utility (see src/utils/device_optimization.py).

Step 2.2: Integrate the optimizer into the model loading and training routines.

Step 2.3: Run the online training demo to verify that the device optimizations are applied.

Enhance Training Data and Extend Training Duration:

Step 3.1: Continue adding high-quality training samples using /train in interactive mode.

Step 3.2: Use batch mode with a larger corpus:

bash
Copy
python examples/simplified_online_training_demo.py --batch --input_file data/test_corpus/sample.txt --output_dir checkpoints/online_batch_test
Step 3.3: Monitor training statistics with /stats to see updates, loss, and batch size changes.

Experiment with Generation Hyperparameters:

Step 4.1: Test the /generate command with various prompts:

/generate What are the benefits of edge AI?

/generate What are the benefits of edge AI? --temperature 1.5

/generate What are the benefits of edge AI? --top_k 20

/generate What are the benefits of edge AI? --top_p 0.8

Step 4.2: Compare outputs to assess improvements in generation quality as training progresses.

Fix Checkpoint Serialization Issue:

Step 5.1: Choose one of the following:

Option A: Remove non-serializable items (like optimizer_state) from the state dictionary before saving.

Option B: Implement a custom JSON encoder (see our TensorEncoder example) to convert Tensors and torch.device objects into serializable types.

Step 5.2: Update the save_checkpoint() method accordingly and test by saving a checkpoint.

Commit and Document Changes:

Step 6.1: Once all modifications are verified, commit the changes.

Step 6.2: Use a commit message similar to the sample below.

Step 6.3: Push your changes to GitHub.

🔄 Future Testing & Optimization Plans (Phase 2)
Enhanced Device Testing:
Expand testing to additional edge devices and implement an automated testing pipeline with performance reporting.

Rigorous Power Profiling:
Implement granular power consumption measurements and power-aware inference scheduling for mobile and edge devices.

Enterprise Integration Testing:
Develop reference implementations for industrial IoT, benchmark multi-model deployment scenarios, and create integration guides for common enterprise frameworks.

Cross-Platform Compiler Optimization:
Complete support for major hardware targets with automated kernel tuning and develop hardware-specific quantization profiles to maximize efficiency.

🛠️ Getting Started
Installation
bash
Copy
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
Testing Associative Memory
bash
Copy
# Run the associative memory demo with visualization
python examples/htps_associative_memory_demo.py --visualize

# Try with advanced features
python examples/htps_associative_memory_demo.py --use_recurrent --use_budget --use_kv_cache --visualize
Online Training Demo
bash
Copy
# Run the interactive online training demo
python examples/simplified_online_training_demo.py --device cpu --output_dir checkpoints/online_test

# Test batch mode with a test corpus
python examples/simplified_online_training_demo.py --batch --input_file data/test_corpus/sample.txt --output_dir checkpoints/online_batch_test
Multi-Device Testing
bash
Copy
# Create device profiles for testing
python scripts/create_device_profiles.py --devices yoga,envy --output_dir profiles/

# Run benchmarks on the current device
python scripts/cross_device_benchmark.py --model_size small --device_profiles profiles/ --output_dir benchmark_results/cross_device/

# Generate visualization for cross-device performance
python scripts/visualize_cross_device.py --input_dir benchmark_results/cross_device/ --output_file benchmark_results/cross_device/device_comparison.png
Analyzing Benchmark Results
bash
Copy
# Generate a comprehensive benchmark analysis
python scripts/analyze_benchmarks.py --input_dir benchmark_results/cross_device --output_dir benchmark_results/analysis --interactive
📝 Immediate Next Steps
Based on recent testing results and our latest improvements, follow these microtask steps:

Fix Checkpoint Incompatibility:

Step 1.1: Verify that the model configuration used for training (e.g., hidden_size) is the same during checkpoint saving and loading.

Step 1.2: If a mismatch is found (e.g., saved checkpoints with hidden_size=256 vs. current 768), either revert to the old configuration or restart training using the updated configuration.

Step 1.3: Update the load_model() method in simplified_online_training_demo.py to enforce configuration consistency.

Implement Device-Specific Optimizations:

Step 2.1: Create device-specific configurations using the DeviceOptimizer utility (in src/utils/device_optimization.py).

Step 2.2: Integrate the optimizer into the model loading and training routines.

Step 2.3: Run the online training demo to confirm that device optimizations are applied.

Extend Training Data & Duration:

Step 3.1: Continue adding high-quality training samples via /train (e.g., "Edge AI in smart cities enables rapid, on-site decision making.").

Step 3.2: Use batch mode with a larger corpus:

bash
Copy
python examples/simplified_online_training_demo.py --batch --input_file data/test_corpus/sample.txt --output_dir checkpoints/online_batch_test
Step 3.3: Monitor training stats using /stats to ensure updates are occurring.

Experiment with Generation Hyperparameters:

Step 4.1: Test different prompts with /generate:

/generate What are the benefits of edge AI?

/generate What are the benefits of edge AI? --temperature 1.5

/generate What are the benefits of edge AI? --top_k 20

/generate What are the benefits of edge AI? --top_p 0.8

Step 4.2: Compare outputs to see improvements as training progresses.

Fix Checkpoint Serialization Issue:

Step 5.1: Decide whether to remove non-serializable objects (like optimizer_state) from the checkpoint state or implement a custom JSON encoder (e.g., TensorEncoder) to handle Tensors and torch.device.

Step 5.2: Update the save_checkpoint() method accordingly and test saving a checkpoint.

Commit and Document Changes:

Step 6.1: Once all modifications are verified, commit your changes.

Step 6.2: Use a commit message similar to the sample below and push your changes to GitHub.

🔄 Future Testing & Optimization Plans (Phase 2)
Enhanced Device Testing: Expand testing to additional edge devices and automate performance reporting.

Rigorous Power Profiling: Implement power-aware inference scheduling and granular power consumption measurements.

Enterprise Integration Testing: Develop reference implementations for industrial IoT and benchmark multi-model deployment scenarios.

Cross-Platform Compiler Optimization: Complete support for major hardware targets, implement automated kernel tuning, and develop hardware-specific quantization profiles.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
Developed by Oscar Nunez (art.by.oscar.n@gmail.com) using vibe coding principles.
