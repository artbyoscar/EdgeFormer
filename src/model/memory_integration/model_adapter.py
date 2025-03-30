"""Adapter for integrating associative memory with EdgeFormer models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MemoryModelAdapter:
    """Integrates EdgeFormer model with HTPS associative memory."""
    
    def __init__(self, model, memory, retriever, device="cpu"):
        """
        Initialize the adapter.
        
        Args:
            model: EdgeFormer model
            memory: HTPSMemory instance
            retriever: MemoryRetriever instance
            device: Device to run on
        """
        self.model = model
        self.memory = memory
        self.retriever = retriever
        self.device = device
        
        # For visualization
        self.attention_maps = []
        self.retrieved_memories = []
        
        logger.info("Initialized MemoryModelAdapter")
    
    def forward(self, input_ids, attention_mask=None, capture_attention=False):
        """
        Forward pass with memory integration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            capture_attention: Whether to capture attention weights
            
        Returns:
            Model outputs with memory integration
        """
        # Reset visualization data
        if capture_attention:
            self.attention_maps = []
            self.retrieved_memories = []
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        logits, hidden_states = outputs[0], outputs[1][-1]  # Last layer's hidden states
        
        # Retrieve memories based on the last token's representation
        last_token_hidden = hidden_states[:, -1:, :]
        
        # Retrieve relevant memories
        memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
            last_token_hidden, self.memory, top_k=3, capture_attention=capture_attention
        )
        
        # Store for visualization
        if capture_attention and attention_weights is not None:
            self.attention_maps.append(attention_weights)
            self.retrieved_memories.append(memory_texts)
        
        # If no memories retrieved, return original outputs
        if memory_vectors is None or len(memory_vectors) == 0:
            return outputs
        
        # Integrate memory influence into the model's hidden states
        memory_vector = torch.matmul(attention_weights, memory_vectors)
        
        # Project to vocabulary space using the model's language modeling head
        if hasattr(self.model, 'lm_head'):
            memory_logits = self.model.lm_head(memory_vector)
            
            # Combine with original logits (weighted sum)
            memory_weight = 0.2  # How much to weight the memory influence
            updated_logits = logits.clone()
            updated_logits[:, -1, :] = (1 - memory_weight) * logits[:, -1, :] + memory_weight * memory_logits.squeeze(1)
            
            # Return updated outputs
            return (updated_logits,) + outputs[1:]
        
        return outputs
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        attention_mask=None,
        use_recurrent=False,
        min_iterations=2,
        max_iterations=8,
        convergence_threshold=0.005,
        capture_attention=False,
    ):
        """
        Generate text with memory integration.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            attention_mask: Attention mask
            use_recurrent: Whether to use recurrent processing
            min_iterations: Minimum recurrent iterations
            max_iterations: Maximum recurrent iterations
            convergence_threshold: Threshold for convergence
            capture_attention: Whether to capture attention weights
            
        Returns:
            Generated token IDs
        """
        # Reset visualization data
        if capture_attention:
            self.attention_maps = []
            self.retrieved_memories = []
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length - input_ids.size(1)):
            # Get model outputs
            if past_key_values is None:
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                logits, hidden_states = outputs[0], outputs[1][-1]
            else:
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask[:, -1:] if attention_mask is not None else None,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                logits, hidden_states = outputs[0], outputs[1][-1]
            
            next_token_logits = logits[:, -1, :]
            
            # Get past key values for next iteration
            if len(outputs) > 1 and isinstance(outputs[-1], tuple):
                past_key_values = outputs[-1]
            
            # Get the last token's hidden state
            last_token_hidden = hidden_states[:, -1:, :]
            
            # Retrieve and integrate memories
            memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
                last_token_hidden, self.memory, top_k=3, capture_attention=capture_attention
            )
            
            # Store for visualization
            if capture_attention and attention_weights is not None:
                self.attention_maps.append(attention_weights)
                self.retrieved_memories.append(memory_texts)
            
            # Integrate with next token logits if memories were retrieved
            if memory_vectors is not None and attention_weights is not None and len(memory_vectors) > 0:
                # Create memory vector
                memory_vector = torch.matmul(attention_weights, memory_vectors)
                
                # Project to vocabulary space
                if hasattr(self.model, 'lm_head'):
                    memory_logits = self.model.lm_head(memory_vector).squeeze(1)
                    
                    # Combine with original logits (weighted sum)
                    memory_weight = 0.2  # How much to weight the memory
                    next_token_log