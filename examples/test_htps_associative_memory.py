#!/usr/bin/env python
# EdgeFormer - HTPS Associative Memory Test Script with Profiling
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

"""
This script tests the HTPS associative memory components with profiling to 
optimize performance. It analyzes memory retrieval efficiency, memory selection
strategies, and computational overhead across different configurations.
"""

import os
import sys
import argparse
import time
import cProfile
import pstats
import io
import logging
from pathlib import Path
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.associative_memory.htps_memory import HTPSMemory
from src.model.associative_memory.memory_retriever import MemoryRetriever
from src.utils.text_dataset import get_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('memory_test')

class MemoryProfiler:
    """
    Profiles memory operations to optimize performance.
    """
    
    def __init__(self, args):
        self.args = args
        self.hidden_size = 512  # Default embedding dimension
        self.tokenizer = get_tokenizer()
        
        # Initialize memory components
        self.initialize_memory()
        
    def initialize_memory(self):
        """Initialize memory components with different configurations"""
        logger.info("Initializing memory components...")
        
        # Initialize HTPS Memory with different selection strategies
        self.htps_memory = {
            'combined': HTPSMemory(
                embedding_dim=self.hidden_size,
                selection_strategy='combined',
                max_entries=100
            ),
            'importance': HTPSMemory(
                hidden_size=self.hidden_size, 
                selection_strategy='importance',
                max_entries=100
            ),
            'recency': HTPSMemory(
                hidden_size=self.hidden_size, 
                selection_strategy='recency',
                max_entries=100
            ),
            'frequency': HTPSMemory(
                hidden_size=self.hidden_size, 
                selection_strategy='frequency',
                max_entries=100
            )
        }
        
        # Initialize memory retriever
        self.memory_retriever = MemoryRetriever(
            hidden_size=self.hidden_size,
            num_attention_heads=8
        )
        
        logger.info("Memory components initialized")
        
    def generate_test_memories(self, num_memories=20):
        """Generate test memories for profiling"""
        logger.info(f"Generating {num_memories} test memories...")
        
        test_texts = [
            "EdgeFormer is a high-performance Transformer implementation optimized for edge devices.",
            "Multi-Head Latent Attention (MLA) reduces KV cache size by projecting keys and values into a compressed shared latent space.",
            "Grouped-Query Attention (GQA) improves efficiency by sharing key/value heads among groups of query heads.",
            "HyperTree-Inspired Budget Forcing intelligently allocates compute resources during inference.",
            "Value-Based Recurrent Depth Processing enables implicit reasoning in latent space.",
            "LIMO training principles focus on quality over quantity in training examples.",
            "EdgeFormer supports long sequences through optimized attention mechanisms and CPU RAM offloading.",
            "Associative Memory Chains incorporate key information during inference with HTPS-inspired selection.",
            "Continuous Latent Reasoning enables LLM reasoning in continuous latent space.",
            "Zero-Shot Adaptive Computation supports per-token adaptive exits based on KV divergence.",
            "The model architecture is optimized for AMD Ryzen/Radeon systems.",
            "DirectML acceleration options are being explored for AMD GPU support.",
            "INT4/INT8 Quantization achieves significant memory reduction with minimal quality loss.",
            "Memory-Aware Chunking provides adaptive processing strategies for long sequences.",
            "Sliding Window Attention efficiently handles longer sequences by limiting attention scope locally.",
            "FlashAttention integration provides highly optimized standard attention computation.",
            "Cross-Platform Optimization leverages MLIR/TVM/Triton for hardware-specific kernels.",
            "Graph-Enhanced Processing supports graph-structured data with virtual node tokens.",
            "Simplified Online Training Pipeline provides on-device fine-tuning capabilities.",
            "Advanced Quantization Profiles explore INT2/1-bit quantization options."
        ]
        
        # Add more generated examples if needed
        while len(test_texts) < num_memories:
            test_texts.append(f"Additional test memory {len(test_texts) + 1} for EdgeFormer testing.")
        
        # Create mock hidden states
        for strategy in self.htps_memory:
            for text in test_texts:
                # Create random hidden states
                mock_hidden = torch.randn(1, len(text.split()), self.hidden_size)
                
                # Add to memory
                self.htps_memory[strategy].add_memory(text, mock_hidden)
                
        logger.info("Test memories generated")
        
    def profile_memory_retrieval(self, num_iterations=100):
        """Profile memory retrieval performance"""
        logger.info(f"Profiling memory retrieval ({num_iterations} iterations)...")
        
        # Test query
        query = "How does EdgeFormer optimize Transformer inference?"
        query_hidden = torch.randn(1, len(query.split()), self.hidden_size)
        
        results = {}
        
        # Profile each strategy
        for strategy, memory in self.htps_memory.items():
            logger.info(f"Testing '{strategy}' selection strategy...")
            
            # Create profiler
            pr = cProfile.Profile()
            pr.enable()
            
            # Run retrieval multiple times
            start_time = time.time()
            for _ in range(num_iterations):
                # Get memory entries
                entries = memory.get_all_entries()
                
                # Retrieve memories
                retrieved_memories = self.memory_retriever.retrieve(
                    query_hidden, 
                    [entry['hidden_states'] for entry in entries],
                    top_k=3
                )
            
            elapsed = time.time() - start_time
            
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 functions
            
            results[strategy] = {
                'time': elapsed,
                'avg_time': elapsed / num_iterations,
                'profile': s.getvalue()
            }
            
            logger.info(f"Strategy '{strategy}' completed in {elapsed:.4f}s ({elapsed/num_iterations:.6f}s per retrieval)")
        
        # Save results
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            for strategy, data in results.items():
                profile_path = os.path.join(self.args.output_dir, f"profile_{strategy}.txt")
                with open(profile_path, 'w') as f:
                    f.write(f"Strategy: {strategy}\n")
                    f.write(f"Total time: {data['time']:.4f}s\n")
                    f.write(f"Average time per retrieval: {data['avg_time']:.6f}s\n\n")
                    f.write(data['profile'])
            
            logger.info(f"Profile results saved to {self.args.output_dir}")
        
        return results
    
    def profile_memory_selection(self, num_iterations=50):
        """Profile memory selection performance"""
        logger.info(f"Profiling memory selection ({num_iterations} iterations)...")
        
        results = {}
        
        # Profile each strategy
        for strategy, memory in self.htps_memory.items():
            logger.info(f"Testing '{strategy}' selection strategy...")
            
            # Create profiler
            pr = cProfile.Profile()
            pr.enable()
            
            # Run selection multiple times
            start_time = time.time()
            for _ in range(num_iterations):
                # Add a new memory that would trigger selection
                mock_hidden = torch.randn(1, 10, self.hidden_size)
                memory.add_memory(f"Test memory {time.time()}", mock_hidden)
            
            elapsed = time.time() - start_time
            
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 functions
            
            results[strategy] = {
                'time': elapsed,
                'avg_time': elapsed / num_iterations,
                'profile': s.getvalue()
            }
            
            logger.info(f"Strategy '{strategy}' completed in {elapsed:.4f}s ({elapsed/num_iterations:.6f}s per selection)")
        
        # Save results
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            for strategy, data in results.items():
                profile_path = os.path.join(self.args.output_dir, f"selection_profile_{strategy}.txt")
                with open(profile_path, 'w') as f:
                    f.write(f"Strategy: {strategy}\n")
                    f.write(f"Total time: {data['time']:.4f}s\n")
                    f.write(f"Average time per selection: {data['avg_time']:.6f}s\n\n")
                    f.write(data['profile'])
            
            logger.info(f"Selection profile results saved to {self.args.output_dir}")
        
        return results
    
    def profile_memory_integration(self, num_iterations=20):
        """Profile full memory integration performance"""
        logger.info(f"Profiling full memory integration ({num_iterations} iterations)...")
        
        # Use combined strategy for full integration test
        memory = self.htps_memory['combined']
        retriever = self.memory_retriever
        
        # Test queries
        queries = [
            "How does EdgeFormer optimize Transformer inference?",
            "What attention mechanisms does EdgeFormer support?",
            "How does the LIMO training approach work?",
            "What is Value-Based Recurrent Depth Processing?",
            "How does associative memory improve reasoning tasks?"
        ]
        
        # Create profiler
        pr = cProfile.Profile()
        pr.enable()
        
        # Run full integration test
        start_time = time.time()
        for _ in range(num_iterations):
            for query in queries:
                # Create query embedding
                query_hidden = torch.randn(1, len(query.split()), self.hidden_size)
                
                # Get memory entries
                entries = memory.get_all_entries()
                
                # Retrieve memories
                retrieved_memories = retriever.retrieve(
                    query_hidden, 
                    [entry['hidden_states'] for entry in entries],
                    top_k=3
                )
                
                # Simulate memory integration with new information
                mock_hidden = torch.randn(1, 15, self.hidden_size)
                memory.add_memory(f"New information about {query}", mock_hidden)
        
        elapsed = time.time() - start_time
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Print top 30 functions
        
        # Save results
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            profile_path = os.path.join(self.args.output_dir, "full_integration_profile.txt")
            with open(profile_path, 'w') as f:
                f.write("Full Memory Integration Profile\n")
                f.write(f"Total time: {elapsed:.4f}s\n")
                f.write(f"Average time per iteration: {elapsed/num_iterations:.6f}s\n\n")
                f.write(s.getvalue())
            
            logger.info(f"Full integration profile saved to {profile_path}")
        
        logger.info(f"Full integration test completed in {elapsed:.4f}s ({elapsed/(num_iterations*len(queries)):.6f}s per query)")
        
        return {
            'time': elapsed,
            'avg_time_per_query': elapsed / (num_iterations * len(queries)),
            'profile': s.getvalue()
        }
    
    def run_all_profiles(self):
        """Run all profiling tests"""
        logger.info("Starting comprehensive memory profiling...")
        
        # Generate test memories
        self.generate_test_memories(num_memories=self.args.num_memories)
        
        # Run all profiles
        retrieval_results = self.profile_memory_retrieval(num_iterations=self.args.num_iterations)
        selection_results = self.profile_memory_selection(num_iterations=self.args.num_iterations // 2)
        integration_results = self.profile_memory_integration(num_iterations=self.args.num_iterations // 5)
        
        # Generate summary report
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            summary_path = os.path.join(self.args.output_dir, "memory_profiling_summary.md")
            with open(summary_path, 'w') as f:
                f.write("# HTPS Associative Memory Profiling Summary\n\n")
                
                f.write("## Memory Retrieval Performance\n\n")
                f.write("| Strategy | Total Time (s) | Avg Time Per Retrieval (s) |\n")
                f.write("|----------|---------------|----------------------------|\n")
                
                for strategy, data in retrieval_results.items():
                    f.write(f"| {strategy} | {data['time']:.4f} | {data['avg_time']:.6f} |\n")
                
                f.write("\n## Memory Selection Performance\n\n")
                f.write("| Strategy | Total Time (s) | Avg Time Per Selection (s) |\n")
                f.write("|----------|---------------|----------------------------|\n")
                
                for strategy, data in selection_results.items():
                    f.write(f"| {strategy} | {data['time']:.4f} | {data['avg_time']:.6f} |\n")
                
                f.write("\n## Full Integration Performance\n\n")
                f.write(f"Total time: {integration_results['time']:.4f}s\n\n")
                f.write(f"Average time per query: {integration_results['avg_time_per_query']:.6f}s\n\n")
                
                f.write("## Recommendations\n\n")
                
                # Find fastest retrieval strategy
                fastest_retrieval = min(retrieval_results.items(), key=lambda x: x[1]['avg_time'])
                f.write(f"* Fastest retrieval strategy: **{fastest_retrieval[0]}** ({fastest_retrieval[1]['avg_time']:.6f}s per retrieval)\n")
                
                # Find fastest selection strategy
                fastest_selection = min(selection_results.items(), key=lambda x: x[1]['avg_time'])
                f.write(f"* Fastest selection strategy: **{fastest_selection[0]}** ({fastest_selection[1]['avg_time']:.6f}s per selection)\n")
                
                # Overall recommendations
                f.write("\n### Implementation Recommendations\n\n")
                f.write("1. For computationally constrained scenarios, use the **{0}** selection strategy\n".format(fastest_selection[0]))
                f.write("2. For applications requiring fast retrieval, use the **{0}** retrieval approach\n".format(fastest_retrieval[0]))
                f.write("3. Consider batching memory operations to amortize computational costs\n")
                f.write("4. The estimated computational overhead for associative memory is approximately {0:.2f}ms per query\n".format(integration_results['avg_time_per_query'] * 1000))
            
            logger.info(f"Summary report generated at {summary_path}")
        
        logger.info("Memory profiling completed successfully")


def main():
    parser = argparse.ArgumentParser(description='HTPS Associative Memory Profiling')
    
    parser.add_argument('--profile', action='store_true',
                        help='Run profiling on memory operations')
    parser.add_argument('--output_dir', type=str, default='benchmark_results/memory_profiling',
                        help='Directory to save profiling results')
    parser.add_argument('--num_memories', type=int, default=50,
                        help='Number of test memories to generate')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of iterations for profiling')
    
    args = parser.parse_args()
    
    profiler = MemoryProfiler(args)
    
    if args.profile:
        profiler.run_all_profiles()
    else:
        logger.info("No action specified. Use --profile to run profiling tests.")


if __name__ == "__main__":
    main()