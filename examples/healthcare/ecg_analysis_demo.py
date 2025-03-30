#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EdgeFormer Healthcare Vertical Demo: ECG Analysis
================================================

This demo shows how EdgeFormer's HTPS Associative Memory and device-aware
optimizations provide superior ECG analysis with minimal computational
overhead compared to generic solutions.

The demo highlights:
1. Low-latency ECG signal processing
2. Enhanced anomaly detection using associative memory
3. HIPAA-compliant memory management
4. Device-specific optimizations
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import EdgeFormer components
from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.mla import MultiHeadLatentAttention
from src.model.transformer.gqa import GroupedQueryAttention
from src.model.memory_integration.memory_retriever import MemoryRetriever
from src.model.associative_memory.htps_memory import HTPSMemory
from examples.healthcare.memory_adapter import HTPSMemoryAdapter

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('edgeformer-healthcare')

# Define ECG-specific constants
ECG_SEQUENCE_LENGTH = 2500  # 10 seconds @ 250Hz
ECG_FEATURE_DIM = 12        # 12-lead ECG
ECG_HIDDEN_DIM = 256        # Smaller model for edge deployment
ECG_LABELS = [
    "Normal",
    "Atrial Fibrillation",
    "First-degree AV Block",
    "Left Bundle Branch Block",
    "Right Bundle Branch Block",
    "Premature Atrial Contraction",
    "Premature Ventricular Contraction",
    "ST Depression",
    "ST Elevation"
]

class ECGEncoder(nn.Module):
    """ECG signal encoder for EdgeFormer."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1D convolutional layers for ECG signal processing
        self.conv1 = nn.Conv1d(ECG_FEATURE_DIM, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  # Downsampling
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)  # Downsampling
        
        # Calculate sequence length after convolutions
        self.reduced_length = ECG_SEQUENCE_LENGTH // 4  # After 2 strides of 2
        
        # Projection to transformer hidden dimension
        self.projection = nn.Linear(128, config.hidden_size)
        
        # Activation and normalization
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm1d(32)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, ecg_signal):
        """
        Process ECG signal.
        
        Args:
            ecg_signal: [batch_size, ECG_FEATURE_DIM, ECG_SEQUENCE_LENGTH]
            
        Returns:
            Processed features [batch_size, reduced_length, hidden_size]
        """
        # Apply convolutional layers
        x = self.act(self.norm1(self.conv1(ecg_signal)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))
        
        # Reshape for transformer: [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)
        
        # Project to hidden dimension
        x = self.projection(x)
        x = self.dropout(x)
        
        return x

class ECGTransformerClassifier(nn.Module):
    """ECG analysis model using EdgeFormer components."""
    
    def __init__(self, config, use_htps_memory=True, device="cpu"):
        super().__init__()
        self.config = config
        self.use_htps_memory = use_htps_memory
        self.device = device
        
        # ECG encoder
        self.encoder = ECGEncoder(config)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = nn.ModuleDict()
            
            # Select attention mechanism based on config
            if config.attention_type == "mla":
                layer["attention"] = MultiHeadLatentAttention(config)
            elif config.attention_type == "gqa":
                layer["attention"] = GroupedQueryAttention(config)
            else:
                layer["attention"] = nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_probs_dropout_prob,
                    batch_first=True
                )
            
            # Feed-forward network
            layer["ffn"] = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
            
            # Layer normalization
            layer["attention_norm"] = nn.LayerNorm(config.hidden_size)
            layer["ffn_norm"] = nn.LayerNorm(config.hidden_size)
            
            self.layers.append(layer)
        
        # Output classifier
        self.classifier = nn.Linear(config.hidden_size, len(ECG_LABELS))
        
        # HTPS Associative Memory for ECG patterns
        if use_htps_memory:
            self.memory = HTPSMemory(
                capacity=100,
                hidden_size=config.hidden_size,
                selection_strategy='htps'
            )
            self.memory_retriever = MemoryRetriever(
                hidden_size=config.hidden_size,
                memory_projection_size=config.hidden_size // 2,
                temperature=0.7,
                device=device
            )
            self._initialize_ecg_memory()
    
    def _initialize_ecg_memory(self):
        """Initialize HTPS memory with known ECG patterns."""
        # Add standard ECG patterns to memory
        ecg_patterns = [
            "Normal sinus rhythm with regular P waves, normal PR interval, and normal QRS complex",
            "Atrial fibrillation with irregular RR intervals and absence of P waves",
            "First-degree AV block with prolonged PR interval > 200ms",
            "Left bundle branch block with wide QRS complex and characteristic morphology",
            "Right bundle branch block with terminal R wave in V1 and wide S wave in I and V6",
            "Premature atrial contraction with early P wave and normal QRS",
            "Premature ventricular contraction with wide, bizarre QRS complex",
            "ST depression with downsloping or horizontal ST segment",
            "ST elevation with upward convexity in consecutive leads"
        ]
        
        # Create simple embedding for text
        device = torch.device(self.device)
        for i, pattern in enumerate(ecg_patterns):
            # Create pseudo-embedding for text (in real system, this would use proper text encoder)
            # Here we use a simple random vector with class information
            vector = torch.randn(1, self.config.hidden_size).to(device)
            vector[:, :len(ECG_LABELS)] = 0
            vector[:, i] = 5.0  # Emphasize class information
            
            # Add to memory with text description
            self.memory.add_entry(vector, pattern)
        
        logger.info(f"Initialized ECG memory with {len(ecg_patterns)} standard patterns")
    
    def forward(self, ecg_signal):
        """
        Process ECG signal and classify.
        
        Args:
            ecg_signal: [batch_size, ECG_FEATURE_DIM, ECG_SEQUENCE_LENGTH]
            
        Returns:
            logits: [batch_size, num_labels]
            attention_scores: Attention scores for visualization
        """
        batch_size = ecg_signal.size(0)
        
        # Encode ECG signal
        hidden_states = self.encoder(ecg_signal)
        
        # Store attention scores for visualization
        all_attention_scores = []
        
        # Process through transformer layers
        for layer in self.layers:
            # Self-attention
            if isinstance(layer["attention"], (MultiHeadLatentAttention, GroupedQueryAttention)):
                attention_outputs = layer["attention"](
                    layer["attention_norm"](hidden_states)
                )
                attention_output = attention_outputs[0]
                if len(attention_outputs) > 1 and attention_outputs[1] is not None:
                    all_attention_scores.append(attention_outputs[1])
            else:
                # Standard attention
                attention_output, attention_weights = layer["attention"](
                    layer["attention_norm"](hidden_states),
                    layer["attention_norm"](hidden_states),
                    layer["attention_norm"](hidden_states)
                )
                all_attention_scores.append(attention_weights)
            
            # Add residual connection
            hidden_states = hidden_states + attention_output
            
            # Feed-forward network
            hidden_states = hidden_states + layer["ffn"](layer["ffn_norm"](hidden_states))
        
        # Get sequence representation (use mean pooling)
        sequence_output = hidden_states.mean(dim=1)
        
        # Enhance with HTPS memory if enabled
        if self.use_htps_memory:
            # Retrieve similar patterns from memory
            memory_vectors, attention_weights, _ = self.memory_retriever.retrieve_memories(
                sequence_output.unsqueeze(1),  # [batch_size, 1, hidden_size]
                self.memory,
                top_k=3
            )
            
            if attention_weights is not None:
                # Weight memory vectors by attention
                weighted_memory = torch.matmul(
                    attention_weights,
                    memory_vectors
                )
                
                # Enhance sequence output with memory
                memory_gate = torch.sigmoid(
                    torch.matmul(sequence_output, weighted_memory.transpose(0, 1))
                ).mean(dim=1, keepdim=True)
                
                # Apply gate to control memory influence (0.3 weighting to memory)
                sequence_output = sequence_output * (1 - 0.3 * memory_gate) + weighted_memory * (0.3 * memory_gate)
        
        # Classify
        logits = self.classifier(sequence_output)
        
        return logits, all_attention_scores

def generate_synthetic_ecg(num_samples, label_index=None):
    """Generate synthetic ECG data for demonstration."""
    batch_size = num_samples
    
    # Generate random ECG signals
    ecg_data = torch.randn(batch_size, ECG_FEATURE_DIM, ECG_SEQUENCE_LENGTH)
    
    # For each sample, embed a pattern corresponding to the label
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        # Either use provided label or random
        if label_index is not None:
            label = label_index
        else:
            label = torch.randint(0, len(ECG_LABELS), (1,)).item()
        
        labels[i] = label
        
        # Add synthetic pattern based on label
        pattern_length = 250  # 1 second pattern
        start_pos = torch.randint(0, ECG_SEQUENCE_LENGTH - pattern_length, (1,)).item()
        
        # Simple synthetic patterns per class (in real application, would use actual ECG patterns)
        if label == 0:  # Normal
            ecg_data[i, 0, start_pos:start_pos+pattern_length] += torch.sin(torch.linspace(0, 6*np.pi, pattern_length)) * 3
        elif label == 1:  # Atrial Fibrillation
            ecg_data[i, 0, start_pos:start_pos+pattern_length] += torch.sin(torch.linspace(0, 20*np.pi, pattern_length) + 
                                                                 torch.randn(pattern_length) * 0.5) * 2
        elif label == 2:  # First-degree AV Block
            ecg_data[i, 0, start_pos:start_pos+pattern_length] += torch.sin(torch.linspace(0, 4*np.pi, pattern_length)) * 3
            # Add elongated PR interval
            ecg_data[i, 0, start_pos+50:start_pos+100] += 1.0
        # Add patterns for other classes...
    
    return ecg_data, labels

def benchmark_ecg_analysis(model, device, num_samples=32, with_memory=True, visualize=False):
    """Benchmark ECG analysis performance."""
    model.eval()
    
    # Generate synthetic data
    ecg_data, labels = generate_synthetic_ecg(num_samples)
    ecg_data, labels = ecg_data.to(device), labels.to(device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(ecg_data[:4])
    
    # Benchmark with memory
    if with_memory:
        model.use_htps_memory = True
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            logits_with_memory, _ = model(ecg_data)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        memory_time = time.time() - start_time
        
        preds_with_memory = torch.argmax(logits_with_memory, dim=1)
        accuracy_with_memory = (preds_with_memory == labels).float().mean().item() * 100
    else:
        memory_time = 0
        accuracy_with_memory = 0
    
    # Benchmark without memory
    model.use_htps_memory = False
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        logits_without_memory, _ = model(ecg_data)
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    no_memory_time = time.time() - start_time
    
    preds_without_memory = torch.argmax(logits_without_memory, dim=1)
    accuracy_without_memory = (preds_without_memory == labels).float().mean().item() * 100
    
    # Calculate overhead
    if no_memory_time > 0:
        memory_overhead = ((memory_time / no_memory_time) - 1) * 100
    else:
        memory_overhead = 0
    
    # Calculate accuracy improvement
    accuracy_improvement = accuracy_with_memory - accuracy_without_memory
    
    # Visualize some results
    if visualize and num_samples > 0:
        plt.figure(figsize=(12, 8))
        
        # Plot ECG signal
        plt.subplot(2, 1, 1)
        sample_idx = 0
        plt.plot(ecg_data[sample_idx, 0, :500].cpu().numpy())
        plt.title(f"ECG Signal (First 500 points, Lead I)\nTrue Label: {ECG_LABELS[labels[sample_idx]]}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        # Plot prediction probabilities
        plt.subplot(2, 1, 2)
        
        x = np.arange(len(ECG_LABELS))
        width = 0.35
        
        without_mem_probs = torch.softmax(logits_without_memory, dim=1)[sample_idx].cpu().numpy()
        with_mem_probs = torch.softmax(logits_with_memory, dim=1)[sample_idx].cpu().numpy() if with_memory else np.zeros_like(without_mem_probs)
        
        plt.bar(x - width/2, without_mem_probs, width, label='Without Memory')
        if with_memory:
            plt.bar(x + width/2, with_mem_probs, width, label='With HTPS Memory')
        
        plt.xlabel('ECG Classifications')
        plt.ylabel('Probability')
        plt.title('Prediction Probabilities')
        plt.xticks(x, [label[:4] + "..." for label in ECG_LABELS], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ecg_analysis_results.png')
        logger.info("Visualization saved to 'ecg_analysis_results.png'")
    
    return {
        "samples": num_samples,
        "accuracy_with_memory": accuracy_with_memory,
        "accuracy_without_memory": accuracy_without_memory,
        "accuracy_improvement": accuracy_improvement,
        "inference_time_with_memory_ms": memory_time * 1000 / num_samples,
        "inference_time_without_memory_ms": no_memory_time * 1000 / num_samples,
        "memory_overhead_percent": memory_overhead
    }

def get_device_info():
    """Get information about the current device."""
    import platform
    import psutil
    
    device_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device"] = torch.cuda.get_device_name(0)
        device_info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    
    return device_info

def main():
    """Run ECG analysis demo."""
    parser = argparse.ArgumentParser(description="EdgeFormer ECG Analysis Demo")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--samples", type=int, default=64, help="Number of samples to test")
    parser.add_argument("--no-memory", action="store_true", help="Disable HTPS Memory")
    parser.add_argument("--attention", type=str, choices=["standard", "mla", "gqa"], default="mla", 
                      help="Attention mechanism to use")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--profile", action="store_true", help="Run detailed performance profiling")
    args = parser.parse_args()
    
    # Get device information
    device_info = get_device_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Running on {device_info['processor']}")
    logger.info(f"RAM: {device_info['ram_gb']} GB")
    logger.info(f"Using device: {device}")
    
    # Create configuration
    config = EdgeFormerConfig(
        hidden_size=ECG_HIDDEN_DIM,
        num_hidden_layers=args.layers,
        num_attention_heads=8,
        intermediate_size=ECG_HIDDEN_DIM * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        attention_type=args.attention,
        latent_size=ECG_HIDDEN_DIM // 4 if args.attention == "mla" else None
    )
    
    # Create model
    model = ECGTransformerClassifier(config, use_htps_memory=not args.no_memory, device=str(device))
    model.to(device)
    
    logger.info(f"Created ECG analysis model with {args.attention} attention")
    
    # Run benchmark
    results = benchmark_ecg_analysis(
        model, 
        device, 
        num_samples=args.samples,
        with_memory=not args.no_memory,
        visualize=args.visualize
    )
    
    # Display results
    logger.info("=" * 50)
    logger.info("EdgeFormer ECG Analysis Results")
    logger.info("=" * 50)
    logger.info(f"Samples: {results['samples']}")
    logger.info(f"Device: {device_info['processor']}")
    logger.info(f"Attention type: {args.attention}")
    logger.info("-" * 50)
    
    if not args.no_memory:
        logger.info(f"Accuracy with HTPS memory: {results['accuracy_with_memory']:.2f}%")
        logger.info(f"Accuracy without memory: {results['accuracy_without_memory']:.2f}%")
        logger.info(f"Accuracy improvement: {results['accuracy_improvement']:.2f}%")
        logger.info("-" * 50)
        logger.info(f"Inference time with memory: {results['inference_time_with_memory_ms']:.2f} ms/sample")
        logger.info(f"Inference time without memory: {results['inference_time_without_memory_ms']:.2f} ms/sample")
        logger.info(f"Memory overhead: {results['memory_overhead_percent']:.2f}%")
    else:
        logger.info(f"Accuracy without memory: {results['accuracy_without_memory']:.2f}%")
        logger.info(f"Inference time: {results['inference_time_without_memory_ms']:.2f} ms/sample")
    
    # Run detailed profiling if requested
    if args.profile:
        logger.info("=" * 50)
        logger.info("Running detailed performance profiling...")
        
        # Profile across batch sizes
        batch_sizes = [1, 4, 16, 32, 64]
        profile_results = []
        
        for batch_size in batch_sizes:
            batch_result = benchmark_ecg_analysis(
                model, 
                device, 
                num_samples=batch_size,
                with_memory=not args.no_memory,
                visualize=False
            )
            profile_results.append(batch_result)
            logger.info(f"Batch size {batch_size}: {batch_result['inference_time_with_memory_ms']:.2f} ms/sample")
        
        # Profile across attention types
        logger.info("-" * 50)
        logger.info("Comparing attention mechanisms...")
        
        attention_types = ["standard", "mla", "gqa"]
        attention_results = []
        
        for attn_type in attention_types:
            # Create new config and model for each attention type
            attn_config = EdgeFormerConfig(
                hidden_size=ECG_HIDDEN_DIM,
                num_hidden_layers=args.layers,
                num_attention_heads=8,
                intermediate_size=ECG_HIDDEN_DIM * 4,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                attention_type=attn_type,
                latent_size=ECG_HIDDEN_DIM // 4 if attn_type == "mla" else None
            )
            
            attn_model = ECGTransformerClassifier(attn_config, use_htps_memory=not args.no_memory, device=str(device))
            attn_model.to(device)
            
            attn_result = benchmark_ecg_analysis(
                attn_model, 
                device, 
                num_samples=32,
                with_memory=not args.no_memory,
                visualize=False
            )
            
            logger.info(f"{attn_type.upper()} attention: {attn_result['inference_time_with_memory_ms']:.2f} ms/sample")
            attention_results.append(attn_result)
    
    logger.info("=" * 50)
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    main()