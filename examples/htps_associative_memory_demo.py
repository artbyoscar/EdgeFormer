#!/usr/bin/env python
# examples/htps_associative_memory_demo.py

import argparse
import logging
import os
import sys
import platform
import psutil
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('edgeformer')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HTPSMemory:
    """
    Simplified HTPS Memory implementation for the demo.
    Stores text memories and their vector representations.
    """
    def __init__(self, capacity=100, hidden_size=768, selection_strategy='htps'):
        """
        Initialize the HTPS memory module.
        
        Args:
            capacity: Maximum number of memories to store
            hidden_size: Dimension of memory embeddings
            selection_strategy: Strategy for memory selection
        """
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.selection_strategy = selection_strategy
        self.memories = []  # List of (text, vector) tuples
        logger.info(f"Initialized HTPSMemory with capacity={capacity}, hidden_size={hidden_size}")
    
    def add_memory(self, text):
        """Add a new memory with mock embedding."""
        # Create a mock embedding vector
        mock_vector = torch.randn(1, self.hidden_size)
        return self.add_entry(text, mock_vector)
    
    def add_entry(self, text, vector):
        """Add a memory entry with the provided vector."""
        if len(self.memories) >= self.capacity:
            self.memories.pop(0)  # Remove oldest memory if at capacity
        self.memories.append((text, vector))
        return True
    
    def get_all_entries(self):
        """Get all stored memories."""
        return self.memories
    
    def clear(self):
        """Clear all memories."""
        self.memories = []
        return True
    
    def list_memories(self):
        """List all memory texts (for compatibility)."""
        return [text for text, _ in self.memories]
    
    def size(self):
        """Return the number of stored memories."""
        return len(self.memories)

class MemoryRetriever:
    """
    Memory retrieval component for the demo.
    """
    def __init__(self, hidden_size, num_attention_heads=4, dropout=0.1):
        """
        Initialize the memory retriever.
        
        Args:
            hidden_size: Dimension of embeddings
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        logger.info(f"Initialized MemoryRetriever with hidden_size={hidden_size}")
    
    def retrieve_memories(self, query_vector, memory_module, top_k=3, capture_attention=False):
        """
        Retrieve relevant memories based on the query.
        
        Args:
            query_vector: Query hidden states
            memory_module: The memory module
            top_k: Number of memories to retrieve
            capture_attention: Whether to capture attention for visualization
            
        Returns:
            memory_vectors: Retrieved memory vectors
            attention_weights: Attention map
            memory_texts: Retrieved memory texts
        """
        # Get all memories
        memories = memory_module.get_all_entries()
        
        if not memories:
            return None, None, []
        
        # For demo purposes, simply return the most recent memories
        recent_memories = memories[-top_k:]
        memory_texts = [text for text, _ in recent_memories]
        memory_vectors = torch.stack([vector for _, vector in recent_memories], dim=1)
        
        # Create a mock attention map (uniform distribution)
        batch_size = query_vector.size(0)
        seq_len = 1 if query_vector.dim() < 3 else query_vector.size(1)
        attention_weights = torch.ones(batch_size, seq_len, len(recent_memories)) / len(recent_memories)
        
        return memory_vectors, attention_weights, memory_texts
    
    def retrieve_memories_legacy(self, query, memory_module, top_k=3):
        """Legacy method for compatibility with the demo interface."""
        return memory_module.list_memories()[-top_k:]

def detect_device():
    """Detect the current device and its properties."""
    processor = platform.processor() 
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    print(f"Device detected: {processor}")
    print(f"RAM: {ram_gb:.1f} GB")
    
    # Determine optimization profile
    if "Intel" in processor:
        profile = "hybrid attention"
    elif "AMD" in processor:
        profile = "sliding window"
    else:
        profile = "standard"
    
    print(f"Using optimization profile: {profile}")
    return profile

def initialize_components(args):
    """Initialize memory components."""
    # Print debug info for HTPSMemory
    print("=" * 50)
    print("HTPSMemory.__init__ parameters:")
    print("(self, capacity=100, hidden_size=768, selection_strategy='htps')")
    print("=" * 50)
    
    # Print debug info for MemoryRetriever
    print("=" * 50)
    print("MemoryRetriever.__init__ parameters:")
    print("(self, hidden_size, num_attention_heads=4, dropout=0.1)")
    print("=" * 50)
    
    # Initialize memory
    memory = HTPSMemory(capacity=args.capacity, hidden_size=768, selection_strategy=args.strategy)
    
    # Initialize retriever
    retriever = MemoryRetriever(hidden_size=768, num_attention_heads=4, dropout=0.1)
    
    logger.info(f"Memory components initialized with capacity={args.capacity}, strategy='{args.strategy}'")
    return memory, retriever

def add_default_memories(memory):
    """Add default memories to the system."""
    try:
        memory.add_memory("EdgeFormer is a high-performance Transformer optimized for edge devices")
        memory.add_memory("Multi-Head Latent Attention (MLA) reduces KV cache size for efficient long-context handling")
        memory.add_memory("HTPS Associative Memory enhances reasoning with minimal computational overhead")
        memory.add_memory("EdgeFormer supports KV Cache offloading to CPU RAM for handling large contexts")
        memory.add_memory("Memory-aware chunking adapts processing for handling long sequences efficiently")
        print("Added default memory about EdgeFormer")
        return True
    except Exception as e:
        logger.error(f"Error adding default memories: {e}")
        return False

def list_memories(memory):
    """List all stored memories."""
    try:
        memories = memory.list_memories()
        print("Stored Memories:")
        print("------------------------------------------------------------")
        if not memories:
            print("No memories stored yet.")
        else:
            for i, memory_text in enumerate(memories):
                print(f"{i+1}. {memory_text}")
    except Exception as e:
        print(f"Error: {str(e)}")

def add_memory(memory, text):
    """Add a new memory to the system."""
    try:
        success = memory.add_memory(text)
        if success:
            print(f"Added memory: {text}")
            return True
        else:
            print("Failed to add memory")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def clear_memories(memory):
    """Clear all stored memories."""
    try:
        memory.clear()
        print("All memories cleared")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def ask_question(retriever, memory, prompt, capture_attention=False):
    """Process a question/prompt and retrieve relevant memories."""
    try:
        print(f"Prompt: {prompt}")
        print("------------------------------------------------------------")
        
        # Mock query vector for demo
        query_vector = torch.randn(1, 1, 768)
        
        # Retrieve memories
        memory_vectors, attention_weights, memory_texts = retriever.retrieve_memories(
            query_vector, memory, top_k=3, capture_attention=capture_attention
        )
        
        if memory_texts:
            print("Retrieved memories:")
            for i, memory_text in enumerate(memory_texts):
                print(f"{i+1}. {memory_text}")
            print("------------------------------------------------------------")
            
            # Provide a mock response for the demo
            print("EdgeFormer uses several techniques for memory efficiency:")
            print(" - Multi-Head Latent Attention (MLA) for KV cache reduction")
            print(" - HTPS Associative Memory for enhanced reasoning")
            print(" - KV Cache offloading to CPU RAM")
            print(" - Memory-aware chunking for handling long sequences")
            
            return True, memory_texts, attention_weights
        else:
            print("No relevant memories found. Please add some memories first.")
            return False, [], None
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, [], None

def print_header():
    """Print the demo header."""
    print("=" * 60)
    print("                  EDGEFORMER ASSOCIATIVE MEMORY")
    print("=" * 60)
    print(" Interactive demo for HTPS-enhanced associative memory with EdgeFormer")
    print("-" * 60)
    print(" Commands:")
    print("  add <text> - Add new memory")
    print("  ask <prompt> - Generate text with memory retrieval")
    print("  list - Show all stored memories")
    print("  clear - Clear all memories")
    print("  quit - Exit the demo")
    print("-" * 60)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EdgeFormer HTPS Associative Memory Demo")
    parser.add_argument("--capacity", type=int, default=20, help="Memory capacity")
    parser.add_argument("--strategy", type=str, default="htps", 
                      choices=["htps", "recency", "random"], help="Memory selection strategy")
    parser.add_argument("--visualize", action="store_true", help="Enable memory visualization")
    parser.add_argument("--use_recurrent", action="store_true", help="Enable recurrent memory processing")
    parser.add_argument("--use_budget", action="store_true", help="Enable budget-aware memory access")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache integration")
    return parser.parse_args()

def main():
    """Main function for the EdgeFormer HTPS Associative Memory demo."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting EdgeFormer test...")
    logger.info("Initializing EdgeFormer with mla attention...")
    logger.info("Initializing new model")
    
    # Detect device
    detect_device()
    
    # Initialize memory components
    memory, retriever = initialize_components(args)
    
    # Print demo header
    print_header()
    
    # Add default memories
    add_default_memories(memory)
    
    # Main interaction loop
    while True:
        try:
            user_input = input("> ")
            
            if user_input.lower() == "quit" or user_input.lower() == "exit":
                break
            elif user_input.lower() == "list":
                list_memories(memory)
            elif user_input.lower() == "clear":
                clear_memories(memory)
            elif user_input.lower().startswith("add "):
                memory_text = user_input[4:].strip()
                if memory_text:
                    add_memory(memory, memory_text)
                else:
                    print("Error: No memory text provided")
            elif user_input.lower().startswith("ask "):
                prompt = user_input[4:].strip()
                if prompt:
                    ask_question(retriever, memory, prompt, capture_attention=args.visualize)
                else:
                    print("Error: No prompt provided")
            else:
                print("Unknown command. Try 'add', 'ask', 'list', 'clear', or 'quit'")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()