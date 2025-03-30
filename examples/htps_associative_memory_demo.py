import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import sys
import inspect

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.edgeformer import EdgeFormer, EdgeFormerConfig
from src.model.associative_memory.htps_memory import HTPSMemory
from src.model.associative_memory.memory_retriever import MemoryRetriever
from src.utils.text_dataset import get_tokenizer
from src.utils.checkpoint_fix import load_checkpoint
from src.utils.model_optimizer import optimize_model_for_device

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('edgeformer')

def debug_htps_memory():
    """Print the parameters expected by HTPSMemory.__init__"""
    print("="*50)
    print("HTPSMemory.__init__ parameters:")
    print(inspect.signature(HTPSMemory.__init__))
    print("="*50)

def debug_memory_retriever():
    """Print the parameters expected by MemoryRetriever.__init__"""
    print("="*50)
    print("MemoryRetriever.__init__ parameters:")
    print(inspect.signature(MemoryRetriever.__init__))
    print("="*50)

def debug_kv_cache_manager():
    """Print the parameters expected by KVCacheManager.__init__"""
    from src.utils.kv_cache_manager import KVCacheManager
    print("="*50)
    print("KVCacheManager.__init__ parameters:")
    print(inspect.signature(KVCacheManager.__init__))
    print("="*50)

class ModelAdapter:
    """
    Adapter class to integrate EdgeFormer model with associative memory.
    """
    def __init__(self, model, memory, retriever, device="cpu"):
        """
        Initialize the model adapter.
        
        Args:
            model: The EdgeFormer model
            memory: The HTPSMemory instance
            retriever: The MemoryRetriever instance
            device: Device to run on (cpu|cuda)
        """
        self.model = model
        self.memory = memory
        self.retriever = retriever
        self.device = device
        
        # For storing visualization data
        self.attention_maps = []
        self.retrieved_memories = []
        
        logger.info("ModelAdapter initialized with EdgeFormer model and associative memory")
    
    def forward(self, input_ids, attention_mask=None, capture_attention=False):
        """
        Forward pass with memory integration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            capture_attention: Whether to capture attention maps for visualization
        
        Returns:
            Model outputs with memory integration
        """
        batch_size, seq_length = input_ids.shape
        
        # Clear visualization data if capturing
        if capture_attention:
            self.attention_maps = []
            self.retrieved_memories = []
        
        # Get initial model outputs
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # Get hidden states from the last layer
        if 'hidden_states' in outputs:
            hidden_states = outputs['hidden_states'][-1]
        else:
            # If hidden_states not included in output, get from forward pass
            _, all_hidden_states = self.model.forward_with_hidden_states(input_ids)
            hidden_states = all_hidden_states[-1]
        
        # Retrieve relevant memories
        memory_vectors, attention_map, memory_texts = self.retriever.retrieve_memories(
            hidden_states, self.memory, capture_attention=capture_attention
        )
        
        # Store for visualization if capturing
        if capture_attention and attention_map is not None:
            self.attention_maps.append(attention_map)
            self.retrieved_memories.append(memory_texts)
        
        # If no memories retrieved, return original outputs
        if memory_vectors is None or memory_vectors.shape[1] == 0:
            return outputs
        
        # Integrate memory with hidden states
        # Simple integration: add memory vectors to hidden states with attention weights
        integrated_hidden_states = hidden_states + torch.matmul(attention_map, memory_vectors)
        
        # Update logits with integrated hidden states
        updated_logits = self.model.lm_head(integrated_hidden_states)
        
        # Update outputs with new logits
        outputs['logits'] = updated_logits
        
        return outputs
    
    def generate(self, input_ids, max_length=100, min_length=0, do_sample=True, 
                 temperature=0.7, top_k=0, top_p=0.9, repetition_penalty=1.0,
                 pad_token_id=None, bos_token_id=None, eos_token_id=None,
                 use_recurrent=False, min_iterations=2, max_iterations=8, 
                 convergence_threshold=0.005, capture_attention=False):
        """
        Generate text with memory integration.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            min_length: Minimum generation length
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            use_recurrent: Whether to use recurrent processing
            min_iterations: Minimum recurrent iterations
            max_iterations: Maximum recurrent iterations
            convergence_threshold: Convergence threshold for recurrent processing
            capture_attention: Whether to capture attention maps for visualization
        
        Returns:
            Generated token IDs
        """
        # Set token IDs from model config if not provided
        if pad_token_id is None:
            pad_token_id = self.model.config.pad_token_id
        if bos_token_id is None:
            bos_token_id = self.model.config.bos_token_id
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id
        
        # Initialize generated sequence with input_ids
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        
        # Clear visualization data if capturing
        if capture_attention:
            self.attention_maps = []
            self.retrieved_memories = []
        
        # Generate tokens one at a time
        for _ in range(max_length - input_ids.shape[1]):
            # Get model outputs with memory integration
            if use_recurrent and self.model.config.enable_recurrent_depth:
                # Initialize model outputs
                with torch.no_grad():
                    outputs = self.forward(generated_ids, capture_attention=capture_attention)
                    logits = outputs['logits']
                    next_token_logits = logits[:, -1, :]
                    
                    # Get hidden states for the last token
                    if 'hidden_states' in outputs:
                        last_hidden = outputs['hidden_states'][-1][:, -1:, :].clone()
                    else:
                        # If hidden_states not included in output, get from forward pass
                        _, all_hidden_states = self.model.forward_with_hidden_states(generated_ids)
                        last_hidden = all_hidden_states[-1][:, -1:, :].clone()
                    
                    current_hidden = last_hidden.clone()
                    
                    # Recurrent processing loop
                    iterations = 0
                    prev_hidden = current_hidden.clone()
                    
                    # Check if the model has value estimator
                    has_value_estimator = hasattr(self.model, 'value_estimator')
                    
                    # Run at least min_iterations iterations
                    while iterations < max_iterations:
                        # Pass through the last transformer layer again
                        current_hidden = self.model.layers[-1].forward(current_hidden)[0]
                        
                        # Check convergence after min_iterations
                        iterations += 1
                        if iterations >= min_iterations:
                            # Check convergence using hidden state difference
                            change = torch.norm(current_hidden - prev_hidden) / torch.norm(prev_hidden)
                            if change < convergence_threshold:
                                break
                        
                        prev_hidden = current_hidden.clone()
                    
                    # After recurrent processing, integrate with memory again
                    memory_vectors, attention_map, memory_texts = self.retriever.retrieve_memories(
                        current_hidden, self.memory, capture_attention=capture_attention
                    )
                    
                    # Store for visualization if capturing
                    if capture_attention and attention_map is not None:
                        self.attention_maps.append(attention_map)
                        self.retrieved_memories.append(memory_texts)
                    
                    # Integrate memory with hidden states if memories were retrieved
                    if memory_vectors is not None and memory_vectors.shape[1] > 0:
                        current_hidden = current_hidden + torch.matmul(attention_map, memory_vectors)
                    
                    # Get logits from the improved hidden state
                    improved_logits = self.model.lm_head(current_hidden)
                    next_token_logits = improved_logits.view(batch_size, -1)
            else:
                # Standard forward pass with memory integration
                with torch.no_grad():
                    outputs = self.forward(generated_ids, capture_attention=capture_attention)
                    logits = outputs['logits']
                    next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for b in range(batch_size):
                    for token_id in generated_ids[b]:
                        next_token_logits[b, token_id] /= repetition_penalty
            
            # Apply top-k/top-p filtering
            if do_sample:
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create scatter indices
                    scatter_indices = sorted_indices.clone()
                    
                    # Apply scattering
                    for b in range(batch_size):
                        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                        next_token_logits[b, indices_to_remove] = -float('inf')
                
                # Top-k sampling
                if top_k > 0:
                    # Remove all tokens with a probability less than the last token of the top-k
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Append the new token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Stop if all sequences have generated the EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated_ids

class AssociativeMemoryDemo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Get tokenizer
        self.tokenizer = get_tokenizer()
        
        # Initialize EdgeFormer model with configuration
        self.config = EdgeFormerConfig(
            vocab_size=50257,  # GPT-2 compatible
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            attention_type=args.attention_type,
            max_position_embeddings=2048,
            enable_budget_forcing=args.use_budget,
            max_budget_tokens=args.max_budget_tokens,
            max_thinking_extensions=args.extensions,
            extension_token="Wait",
            budget_criteria="balanced",
            enable_recurrent_depth=args.use_recurrent,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold
        )
        
        # Initialize model
        logger.info(f"Initializing EdgeFormer with {args.attention_type} attention...")
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"Loading model from {args.model_path}")
            # Use the checkpoint_fix loader instead of from_pretrained
            checkpoint = load_checkpoint(args.model_path, self.device)
    
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                # If checkpoint contains config, use that instead
                logger.info(f"Using configuration from checkpoint")
                model_config = checkpoint['config']
                self.model = EdgeFormer(model_config)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # If no config or just a state dict
                logger.info(f"Using provided configuration with checkpoint weights")
                self.model = EdgeFormer(self.config)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume it's a raw state dict
                    self.model.load_state_dict(checkpoint)
        else:
            logger.info("Initializing new model")
            self.model = EdgeFormer(self.config)

        self.model.to(self.device)
        self.model.eval()
        
        # Add this line here
        self.model = optimize_model_for_device(self.model, self.config)
        
        # Initialize KV Cache Manager if requested
        if args.use_kv_cache:
            from src.utils.kv_cache_manager import KVCacheManager
            # Debug KV Cache Manager parameters
            debug_kv_cache_manager()
            kv_cache_manager = KVCacheManager(
                max_batch_size=1,
                max_seq_length=1024,
                num_layers=self.config.num_hidden_layers,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                device=self.device,
                enable_offload=True
                # Removed offload_threshold parameter
            )
            self.model.kv_cache_manager = kv_cache_manager
            logger.info(f"KV Cache Manager initialized")
        
        # Debug HTPSMemory parameters
        debug_htps_memory()
        # Debug MemoryRetriever parameters
        debug_memory_retriever()
        
        # Initialize associative memory
        self.memory = HTPSMemory(
            capacity=args.memory_size,  # Using capacity instead of memory_size
            hidden_size=self.config.hidden_size,  # Use the model's hidden size from config
            selection_strategy='htps'
        )
        
        # Initialize memory retriever - adjust parameters based on debug output
        self.retriever = MemoryRetriever(
            hidden_size=self.config.hidden_size,  # This is required
            num_attention_heads=4,  # Use num_attention_heads instead of num_heads
            dropout=0.1
        )
        
        # Create model adapter for memory integration
        self.model_adapter = ModelAdapter(
            model=self.model,
            memory=self.memory,
            retriever=self.retriever,
            device=self.device
        )
        
        logger.info(f"Associative memory initialized with size {args.memory_size} and strategy '{args.memory_strategy}'")
        
        # Load initial memories if provided
        if args.memory_file and os.path.exists(args.memory_file):
            self.load_memories_from_file(args.memory_file)
    
    def load_memories_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memories = f.read().strip().split('\n\n')
            
            logger.info(f"Loading {len(memories)} memories from {file_path}")
            for memory in memories:
                if memory.strip():
                    self.add_memory(memory)
            logger.info("Memories loaded successfully")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    def add_memory(self, text):
        """Add new memory to the associative memory system"""
        # Tokenize the text
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Generate embedding for the text
        with torch.no_grad():
            # Use forward_with_hidden_states to get hidden states directly
            outputs, all_hidden_states = self.model.forward_with_hidden_states(tokens)
            hidden_states = all_hidden_states[-1]
            
            # For SimpleTokenizer which doesn't have pad_token_id, use another approach
            # Assume all tokens are valid (no padding)
            mask = torch.ones_like(tokens).float()
            memory_vector = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        
        # Add to memory with the text as the key (using add_entry instead of add_memory)
        self.memory.add_entry(text, memory_vector)
        return True
    
    def generate_with_memory(self, prompt, max_length=100):
        """Generate text using the model with associative memory integration"""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        args = self.args
        
        # Setup for visualization if enabled
        attention_maps = []
        retrieved_memories = []
        
        # Generate with memory integration through the adapter
        output_ids = self.model_adapter.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            use_recurrent=args.use_recurrent,
            min_iterations=args.min_iterations,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
            capture_attention=args.visualize
        )
        
        # If visualization is enabled, capture attention maps and retrieved memories
        if args.visualize:
            attention_maps = self.model_adapter.attention_maps
            retrieved_memories = self.model_adapter.retrieved_memories
        
        # Decode the output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return output_text, attention_maps, retrieved_memories
    
    def visualize_memory_retrieval(self, attention_maps, retrieved_memories, output_path="memory_attention.png"):
        """Visualize memory retrieval attention maps"""
        if not attention_maps or not retrieved_memories:
            logger.info("No attention maps or retrieved memories to visualize")
            return
        
        # Create a figure for the attention maps
        plt.figure(figsize=(12, 8))
        
        # Plot each attention map
        for i, (attn_map, memories) in enumerate(zip(attention_maps, retrieved_memories)):
            if i >= 5:  # Limit to 5 steps for clarity
                break
                
            # Create subplot
            plt.subplot(min(5, len(attention_maps)), 1, i+1)
            
            # Get memory texts for labels
            memory_texts = [mem[:30] + "..." if len(mem) > 30 else mem for mem in memories]
            
            # Plot heatmap
            plt.imshow(attn_map.cpu().numpy(), aspect='auto', cmap='viridis')
            plt.colorbar(fraction=0.046, pad=0.04)
            
            # Add labels
            plt.yticks(range(len(memory_texts)), memory_texts, fontsize=8)
            plt.title(f"Step {i+1} Memory Attention")
            
            # If last plot, add x-axis label
            if i == min(4, len(attention_maps)-1):
                plt.xlabel("Query Position")
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Saved memory attention visualization to {output_path}")
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="EdgeFormer Associative Memory Demo")
    
    # Input/output parameters
    parser.add_argument("--prompt", type=str, default="EdgeFormer is", help="Input prompt for generation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu|cuda)")
    parser.add_argument("--memory_file", type=str, default=None, help="File with initial memories")
    
    # Memory parameters
    parser.add_argument("--memory_size", type=int, default=20, help="Maximum number of memories to store")
    parser.add_argument("--memory_strategy", type=str, default="htps", 
                    choices=["importance", "recency", "frequency", "htps"], 
                    help="Memory selection strategy")
    parser.add_argument("--retrieval_strategy", type=str, default="attention", 
                    choices=["similarity", "attention", "hybrid"], 
                    help="Memory retrieval strategy")
    
    # Feature flags
    parser.add_argument("--use_recurrent", action="store_true", help="Enable value-based recurrent processing")
    parser.add_argument("--use_budget", action="store_true", help="Enable HyperTree budget forcing")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache management")
    
    # KV cache parameters
    parser.add_argument("--offload_threshold", type=int, default=1024, help="Token threshold for offloading to RAM")
    
    # Recurrent processing parameters
    parser.add_argument("--min_iterations", type=int, default=2, help="Minimum recurrent iterations")
    parser.add_argument("--max_iterations", type=int, default=8, help="Maximum recurrent iterations")
    parser.add_argument("--convergence_threshold", type=float, default=0.005, help="Convergence threshold")
    
    # Budget forcing parameters
    parser.add_argument("--max_budget_tokens", type=int, default=2048, help="Maximum budget tokens")
    parser.add_argument("--extensions", type=int, default=2, help="Maximum thinking extensions")
    
    # Attention type
    parser.add_argument("--attention_type", type=str, default="mla", 
                    choices=["standard", "mla", "mla_window"], help="Attention type")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    return parser.parse_args()

def print_header():
    print("\n" + "="*60)
    print(" "*18 + "EDGEFORMER ASSOCIATIVE MEMORY")
    print("="*60)
    print(" Interactive demo for HTPS-enhanced associative memory with EdgeFormer")
    print("-"*60)
    print(" Commands:")
    print("  add <text> - Add new memory")
    print("  ask <prompt> - Generate text with memory retrieval")
    print("  list - Show all stored memories")
    print("  clear - Clear all memories")
    print("  quit - Exit the demo")
    print("-"*60 + "\n")

def main():
    args = parse_args()
    logger.info("Starting EdgeFormer test...")
    
    # Initialize the demo
    demo = AssociativeMemoryDemo(args)
    
    # If a prompt is provided directly, run in non-interactive mode
    if args.prompt and args.prompt != "EdgeFormer is":
        print_header()
        
        # Generate with memory
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        
        output, attention_maps, retrieved_memories = demo.generate_with_memory(args.prompt, args.max_length)
        
        print("\nGenerated text:")
        print("-" * 60)
        print(output)
        print("-" * 60)
        
        # Visualize if requested
        if args.visualize and attention_maps:
            demo.visualize_memory_retrieval(attention_maps, retrieved_memories)
        
        return
    
    # Interactive mode
    print_header()
    
    # Add a default memory if none was loaded
    if len(demo.memory.get_all_entries()) == 0:
        default_memory = """EdgeFormer is a high-performance Transformer implementation optimized to run efficiently 
on edge devices with limited compute resources. It features Multi-Head Latent Attention (MLA), 
Grouped-Query Attention (GQA), and HyperTree-inspired budget forcing mechanisms."""
        demo.add_memory(default_memory)
        print("Added default memory about EdgeFormer")
    
    # Interactive loop
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() == "quit" or command.lower() == "exit":
                break
                
            elif command.lower() == "list":
                memories = demo.memory.get_all_entries()
                print("\nStored Memories:")
                print("-" * 60)
                for i, (text, _) in enumerate(memories):
                    print(f"{i+1}. {text[:60]}..." if len(text) > 60 else f"{i+1}. {text}")
                print(f"\nTotal: {len(memories)} memories")
                
            elif command.lower() == "clear":
                demo.memory.clear()
                print("All memories cleared")
                
            elif command.lower().startswith("add "):
                text = command[4:].strip()
                if text:
                    success = demo.add_memory(text)
                    if success:
                        print(f"Memory added successfully! ({demo.memory.size()} memories stored)")
                else:
                    print("Please provide text to add as memory")
                    
            elif command.lower().startswith("ask ") or command.lower().startswith("generate "):
                prompt = command[4:].strip() if command.lower().startswith("ask ") else command[9:].strip()
                if not prompt:
                    prompt = input("Enter your prompt: ").strip()
                
                if prompt:
                    print(f"\nPrompt: {prompt}")
                    print("-" * 60)
                    
                    output, attention_maps, retrieved_memories = demo.generate_with_memory(prompt, args.max_length)
                    
                    print("\nGenerated text:")
                    print("-" * 60)
                    print(output)
                    print("-" * 60)
                    
                    # Visualize if requested
                    if args.visualize and attention_maps:
                        demo.visualize_memory_retrieval(attention_maps, retrieved_memories)
                else:
                    print("Please provide a prompt")
                    
            elif command.lower() == "help":
                print_header()
                
            else:
                print("Unknown command. Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()