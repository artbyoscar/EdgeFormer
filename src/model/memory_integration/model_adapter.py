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
                    next_token_logits = (1 - memory_weight) * next_token_logits + memory_weight * memory_logits
            
            # Apply recurrent processing if enabled
            if use_recurrent and memory_vectors is not None and len(memory_vectors) > 0:
                # Iterative refinement of predictions
                prev_next_token_logits = next_token_logits.clone()
                iterations = 0
                
                while iterations < max_iterations:
                    iterations += 1
                    
                    # Re-retrieve memories with current prediction context
                    current_hidden = last_token_hidden.clone()
                    
                    # Update with current prediction info
                    if hasattr(self.model, 'lm_head'):
                        # Get embedding of predicted token
                        pred_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        pred_embed = self.model.transformer.embeddings.word_embeddings(pred_token)
                        
                        # Blend with current hidden state
                        blend_factor = 0.3
                        current_hidden = (1 - blend_factor) * current_hidden + blend_factor * pred_embed
                    
                    # Re-retrieve with updated context
                    memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
                        current_hidden, self.memory, top_k=3, capture_attention=capture_attention
                    )
                    
                    # Store for visualization
                    if capture_attention and attention_weights is not None:
                        self.attention_maps.append(attention_weights)
                        self.retrieved_memories.append(memory_texts)
                    
                    # Skip iteration if no memories retrieved
                    if memory_vectors is None or len(memory_vectors) == 0:
                        break
                    
                    # Create memory vector
                    memory_vector = torch.matmul(attention_weights, memory_vectors)
                    
                    # Project to vocabulary space
                    if hasattr(self.model, 'lm_head'):
                        memory_logits = self.model.lm_head(memory_vector).squeeze(1)
                        
                        # Update logits with increasing weight in later iterations
                        adaptive_weight = min(0.1 + (iterations * 0.05), 0.4)
                        next_token_logits = (1 - adaptive_weight) * next_token_logits + adaptive_weight * memory_logits
                    
                    # Check for convergence
                    if torch.max(torch.abs(prev_next_token_logits - next_token_logits)) < convergence_threshold:
                        # Log early convergence
                        logger.debug(f"Converged after {iterations} iterations")
                        break
                    
                    prev_next_token_logits = next_token_logits.clone()
                    
                    # Must complete minimum iterations
                    if iterations >= min_iterations:
                        break
            
            # Adjust with temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))[0]
                if indices_to_remove.size(0) > 0:
                    min_thresh = indices_to_remove[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_thresh,
                        torch.full_like(next_token_logits, float("-inf")),
                        next_token_logits,
                    )
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = torch.where(
                    indices_to_remove,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )
            
            # Sample from the filtered distribution
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update generated and attention mask
            generated = torch.cat([generated, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1
                )
        
        return generated
