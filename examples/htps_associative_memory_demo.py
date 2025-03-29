#!/usr/bin/env python
# EdgeFormer - HTPS Associative Memory Demonstration
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer
from src.model.config import EdgeFormerConfig
from src.model_adaptor import EdgeFormerMemoryAdapter
from src.utils.text_dataset import get_tokenizer
from src.model.associative_memory.htps_memory import HTPSMemory
from src.model.associative_memory.memory_retriever import AttentionBasedRetriever

class AssociativeMemoryDemo:
    """
    Demonstration of EdgeFormer with HTPS-inspired associative memory capabilities.
    This demo shows how the model can dynamically use memory to improve reasoning
    and response quality with minimal computational overhead.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer()
        
        # Configure model
        self.config = EdgeFormerConfig(
            vocab_size=50257,  # GPT-2 compatible
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            max_position_embeddings=1024,
            attention_type=args.attention_type,
            use_cache=True,
            sliding_window=args.sliding_window,
            use_recurrent_processing=args.use_recurrent,
            min_iterations=args.min_iterations,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
            use_budget_forcing=args.use_budget,
            extension_likelihood=0.7,
            extension_factor=2,
            use_kv_cache_offloading=args.use_kv_cache
        )
        
        # Initialize model
        self.model = EdgeFormer(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize memory components
        self.memory = HTPSMemory(
            capacity=args.memory_capacity,
            embedding_dim=self.config.hidden_size,
            selection_strategy=args.selection_strategy
        )
        
        self.retriever = AttentionBasedRetriever(
            model_dim=self.config.hidden_size,
            num_heads=4,
            dropout=0.1,
            temperature=args.temperature,
            use_gating=args.use_gating
        )
        
        # Connect memory components to the model
        self.adapter = EdgeFormerMemoryAdapter(self.model, self.memory, self.retriever)
        
        # Initialize memory with pre-defined facts (if provided)
        if args.memory_file and os.path.exists(args.memory_file):
            self.load_memory_from_file(args.memory_file)
        
        # Setup visualization
        if args.visualize:
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
            self.fig.canvas.manager.set_window_title('EdgeFormer Associative Memory Demo')
    
    def load_memory_from_file(self, file_path):
        """Load memory entries from a text file."""
        print(f"Loading memory from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Adding memory entries"):
            line = line.strip()
            if line and not line.startswith('#'):
                # Encode the memory entry
                input_ids = self.tokenizer.encode(line)
                input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Generate embedding for the memory entry
                with torch.no_grad():
                    outputs = self.model(input_tensor, output_hidden_states=True)
                    # Use the last layer's representation
                    embedding = outputs.hidden_states[-1][:, -1, :].squeeze(0)
                
                # Add to memory with high importance for pre-loaded facts
                self.memory.add_entry(embedding.cpu(), line, importance=0.9)
        
        print(f"Successfully added {len(lines)} memory entries.")
    
    def visualize_memory_activation(self, retrieval_scores, retrieved_entries, query_text):
        """Visualize memory retrieval and activation patterns."""
        if not self.args.visualize:
            return
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot retrieval scores
        sorted_indices = torch.argsort(retrieval_scores, descending=True)
        sorted_scores = retrieval_scores[sorted_indices]
        retrieved_texts = [retrieved_entries[i] for i in sorted_indices[:10]]
        
        # Plot scores for top entries
        num_entries = min(10, len(sorted_scores))
        indices = list(range(num_entries))
        
        self.ax1.bar(indices, sorted_scores[:num_entries].cpu().numpy())
        self.ax1.set_xlabel('Memory Entry Index')
        self.ax1.set_ylabel('Retrieval Score')
        self.ax1.set_title('Top Memory Retrievals')
        self.ax1.set_xticks(indices)
        self.ax1.set_xticklabels([f"{i+1}" for i in range(num_entries)], rotation=45)
        
        # Plot activation heatmap
        if self.retriever.last_attention_weights is not None:
            attn = self.retriever.last_attention_weights.cpu().numpy()
            im = self.ax2.imshow(attn[:1, :, :], cmap='viridis')
            self.ax2.set_title('Attention Activation Map')
            self.fig.colorbar(im, ax=self.ax2)
        else:
            self.ax2.text(0.5, 0.5, 'No attention weights available', 
                          horizontalalignment='center', verticalalignment='center')
        
        # Add query information
        self.fig.suptitle(f"Query: {query_text[:50]}{'...' if len(query_text) > 50 else ''}")
        
        # List top retrieved memories
        memory_text = "\n".join([f"{i+1}. {txt[:50]}{'...' if len(txt) > 50 else ''}" 
                                 for i, txt in enumerate(retrieved_texts)])
        
        plt.figtext(0.5, 0.01, f"Top Retrieved Memories:\n{memory_text}", 
                    horizontalalignment='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        self.fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.pause(0.1)
    
    def memory_enhanced_generation(self, prompt, max_length=100):
        """Generate text with memory enhancement."""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Initial forward pass to get the query embedding
        with torch.no_grad():
            outputs = self.model(input_tensor, output_hidden_states=True)
            query_embedding = outputs.hidden_states[-1][:, -1, :].squeeze(0)
        
        # Retrieve relevant memories based on the query
        retrieved_embeddings, retrieval_scores, retrieved_entries = self.memory.retrieve(
            query_embedding, k=self.args.retrieve_top_k
        )
        
        # Visualize memory retrievals
        self.visualize_memory_activation(retrieval_scores, retrieved_entries, prompt)
        
        print("\n=== Memory Retrievals ===")
        for i, (score, entry) in enumerate(zip(retrieval_scores[:5].cpu().tolist(), retrieved_entries[:5])):
            print(f"{i+1}. [{score:.4f}] {entry[:100]}{'...' if len(entry) > 100 else ''}")
        
        # Generate text with memory enhancement
        print("\n=== Generating Response ===")
        
        # Set parameters for generation
        generation_params = {
            "max_length": max_length,
            "min_length": 1,
            "do_sample": self.args.do_sample,
            "temperature": self.args.generation_temperature,
            "top_k": self.args.top_k,
            "top_p": self.args.top_p,
            "repetition_penalty": 1.1,
            "num_return_sequences": 1,
            "use_recurrent_processing": self.args.use_recurrent,
            "min_iterations": self.args.min_iterations,
            "max_iterations": self.args.max_iterations,
            "convergence_threshold": self.args.convergence_threshold,
            "use_budget_forcing": self.args.use_budget,
            "use_kv_cache": self.args.use_kv_cache,
            "use_memory": True,  # Enable memory-enhanced generation
            "retrieved_memory_embeddings": retrieved_embeddings,
            "retrieved_memory_scores": retrieval_scores,
        }
        
        # Measure generation time
        start_time = time.time()
        
        # Generate text
        output_sequences = self.model.generate(
            input_ids=input_tensor,
            **generation_params
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(output_sequences[0].tolist())
        
        # Add the response to memory if it's long enough and significant
        if len(output_sequences[0]) > 20:
            with torch.no_grad():
                output_emb = self.model(output_sequences, output_hidden_states=True).hidden_states[-1][:, -1, :].squeeze(0)
                # Calculate importance based on uniqueness compared to existing memories
                similarity_scores = []
                for emb in self.memory.embeddings:
                    if len(emb) > 0:
                        cosine_sim = torch.cosine_similarity(output_emb.cpu(), torch.tensor(emb), dim=0)
                        similarity_scores.append(cosine_sim.item())
                
                # Higher importance for more unique content
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                importance = 1.0 - avg_similarity
                
                # Add to memory if sufficiently unique
                if importance > 0.3:
                    self.memory.add_entry(
                        output_emb.cpu(), 
                        self.tokenizer.decode(output_sequences[0][-100:].tolist()), 
                        importance=importance
                    )
        
        return generated_text, generation_time, self.model.recurrent_iterations if hasattr(self.model, 'recurrent_iterations') else 0
    
    def run_demo(self):
        """Run the interactive demo."""
        print("\n" + "="*80)
        print(" "*20 + "EdgeFormer Associative Memory Demo")
        print("="*80)
        
        print("\nInitialized with:")
        print(f"- Memory capacity: {self.args.memory_capacity} entries")
        print(f"- Selection strategy: {self.args.selection_strategy}")
        print(f"- Attention type: {self.args.attention_type}")
        print(f"- Device: {self.args.device}")
        print(f"- Recurrent processing: {'Enabled' if self.args.use_recurrent else 'Disabled'}")
        print(f"- Budget forcing: {'Enabled' if self.args.use_budget else 'Disabled'}")
        print(f"- KV cache management: {'Enabled' if self.args.use_kv_cache else 'Disabled'}")
        
        print("\nCurrent memory status:")
        print(f"- Entries in memory: {len(self.memory)}")
        
        # Interactive demo
        while True:
            print("\n" + "-"*80)
            prompt = input("\nEnter your prompt (or 'exit' to quit): ")
            
            if prompt.lower() in ('exit', 'quit', 'q'):
                break
            
            max_length = self.args.max_length
            if prompt.startswith('!length '):
                try:
                    max_length = int(prompt.split(' ')[1])
                    prompt = ' '.join(prompt.split(' ')[2:])
                except:
                    print("Invalid length format. Using default length.")
            
            # Add memory command
            if prompt.startswith('!addmem '):
                memory_text = prompt[8:]
                input_ids = self.tokenizer.encode(memory_text)
                input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor, output_hidden_states=True)
                    embedding = outputs.hidden_states[-1][:, -1, :].squeeze(0)
                
                self.memory.add_entry(embedding.cpu(), memory_text, importance=0.8)
                print(f"Added to memory: {memory_text[:50]}{'...' if len(memory_text) > 50 else ''}")
                continue
            
            # Clear memory command
            if prompt.lower() == '!clearmem':
                self.memory.clear()
                print("Memory cleared.")
                continue
            
            # Show memory command
            if prompt.lower() == '!showmem':
                entries = self.memory.get_all_entries()
                print(f"\nCurrent Memory ({len(entries)} entries):")
                for i, (importance, entry) in enumerate(sorted(entries, key=lambda x: x[0], reverse=True)[:10]):
                    print(f"{i+1}. [{importance:.4f}] {entry[:100]}{'...' if len(entry) > 100 else ''}")
                continue
            
            # Process normal generation request
            if prompt:
                print(f"\nProcessing: {prompt}")
                output, gen_time, iterations = self.memory_enhanced_generation(prompt, max_length)
                
                # Display output
                print("\n=== Generated Output ===")
                print(output)
                
                # Display performance metrics
                print("\n=== Performance Metrics ===")
                print(f"Generation time: {gen_time:.4f} seconds")
                if iterations > 0:
                    print(f"Recurrent iterations: {iterations}")
                print(f"Tokens per second: {max_length / gen_time:.2f}")
    
    def close(self):
        """Clean up resources."""
        if self.args.visualize:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="EdgeFormer Associative Memory Demo")
    
    # Model configuration
    parser.add_argument('--prompt', type=str, default=None, help='Initial prompt for generation')
    parser.add_argument('--attention_type', type=str, default='standard', choices=['standard', 'mla', 'sliding_window'], 
                        help='Attention mechanism to use')
    parser.add_argument('--sliding_window', type=int, default=512, help='Size of sliding window (if used)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cuda/cpu)')
    
    # Memory configuration
    parser.add_argument('--memory_capacity', type=int, default=100, help='Maximum number of memory entries')
    parser.add_argument('--selection_strategy', type=str, default='htps', 
                        choices=['importance', 'recency', 'frequency', 'htps'], 
                        help='Memory selection strategy')
    parser.add_argument('--memory_file', type=str, default=None, help='File with initial memory entries')
    parser.add_argument('--retrieve_top_k', type=int, default=5, help='Number of memories to retrieve')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for memory attention')
    parser.add_argument('--use_gating', action='store_true', help='Use gating mechanism for memory integration')
    
    # Generation parameters
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--do_sample', action='store_true', help='Use sampling for generation')
    parser.add_argument('--generation_temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling parameter')
    
    # Advanced features
    parser.add_argument('--use_recurrent', action='store_true', help='Use recurrent processing')
    parser.add_argument('--min_iterations', type=int, default=2, help='Minimum recurrent iterations')
    parser.add_argument('--max_iterations', type=int, default=10, help='Maximum recurrent iterations')
    parser.add_argument('--convergence_threshold', type=float, default=0.005, help='Convergence threshold')
    parser.add_argument('--use_budget', action='store_true', help='Use HTPS budget forcing')
    parser.add_argument('--use_kv_cache', action='store_true', help='Use KV cache management')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    
    args = parser.parse_args()
    
    # Initialize and run demo
    demo = AssociativeMemoryDemo(args)
    
    # Run with provided prompt or interactive mode
    if args.prompt:
        output, gen_time, iterations = demo.memory_enhanced_generation(args.prompt, args.max_length)
        print("\n=== Generated Output ===")
        print(output)
        print(f"\nGeneration time: {gen_time:.4f} seconds")
        if iterations > 0:
            print(f"Recurrent iterations: {iterations}")
        
        # If visualization is enabled, wait for user to close the plot
        if args.visualize:
            input("\nPress Enter to exit...")
    else:
        # Run interactive demo
        try:
            demo.run_demo()
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
        finally:
            demo.close()


if __name__ == "__main__":
    main()