"""Adapter for integrating associative memory with EdgeFormer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MemoryModelAdapter:
    """Adapter class to integrate EdgeFormer model with associative memory."""
    
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
    
    def _get_hidden_states(self, input_ids, attention_mask=None):
        """Get hidden states from the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get the last layer's hidden states
        if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
            # outputs[1] contains all_hidden_states if output_hidden_states=True
            hidden_states = outputs[1][-1]  # Last layer's hidden states
        else:
            # Fallback to logits if hidden states not available
            hidden_states = outputs[0]
        
        return hidden_states
    
    def add_memory(self, text, tokenizer=None):
        """
        Add a memory from text.
        
        Args:
            text: Text to add as memory
            tokenizer: Tokenizer for encoding text (if needed)
        """
        if not hasattr(self.memory, 'add_memory'):
            logger.error("Memory module does not support add_memory")
            return False
        
        # Handle text input directly if supported
        if hasattr(self.memory, 'add_memory') and not tokenizer:
            return self.memory.add_memory(text)
        
        # Use tokenizer and model to generate embeddings
        if tokenizer:
            # Tokenize and convert to tensor
            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                hidden_states = self._get_hidden_states(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask")
                )
                
                # Average across sequence length
                if inputs.get("attention_mask") is not None:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden_states * mask).sum(1) / mask.sum(1)
                else:
                    pooled = hidden_states.mean(dim=1)
                
                # Add to memory
                if hasattr(self.memory, 'add_entry'):
                    return self.memory.add_entry(text, pooled)
                else:
                    return self.memory.add_memory(text)
        
        return False
    
    def generate_with_memory(
        self,
        input_ids,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        do_sample=True,
        attention_mask=None,
        tokenizer=None,
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
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            attention_mask: Attention mask
            tokenizer: Tokenizer for decoding
            use_recurrent: Whether to use recurrent processing
            min_iterations: Minimum recurrent iterations
            max_iterations: Maximum recurrent iterations
            convergence_threshold: Threshold for convergence
            capture_attention: Whether to capture attention weights
            
        Returns:
            Generated text or token IDs
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
            outputs = self.model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            
            next_token_logits = outputs[0][:, -1, :]
            past_key_values = outputs[-1] if len(outputs) > 1 else None
            
            # Get hidden states for memory retrieval
            if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
                hidden_states = outputs[1][-1][:, -1:, :]  # Last token of last layer
            else:
                hidden_states = None
            
            # Retrieve and integrate memories
            if hidden_states is not None and self.memory is not None and self.retriever is not None:
                memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
                    hidden_states, self.memory, top_k=3, capture_attention=capture_attention
                )
                
                # Store for visualization
                if capture_attention and attention_weights is not None:
                    self.attention_maps.append(attention_weights)
                    self.retrieved_memories.append(memory_texts)
                
                # Integrate with next token logits if memories were retrieved
                if memory_vectors is not None and attention_weights is not None:
                    # Simple integration: average memory vector influence
                    memory_influence = torch.matmul(attention_weights, memory_vectors)
                    
                    # Project to vocabulary space
                    if hasattr(self.model, 'lm_head'):
                        memory_logits = self.model.lm_head(memory_influence.view(-1, hidden_states.size(-1)))
                        
                        # Combine with original logits (weighted sum)
                        memory_weight = 0.2  # How much to weight the memory
                        next_token_logits = (1 - memory_weight) * next_token_logits + memory_weight * memory_logits
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(generated.size(0)):
                    for token_id in generated[i]:
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Filter with top-k and/or top-p
            if do_sample:
                # Top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(next_token_logits.size(0)):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add next token to generated
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    attention_mask.new_ones((attention_mask.size(0), 1))
                ], dim=-1)
            
            # Check if we've generated EOS token
            if (next_token == self.model.config.eos_token_id).all():
                break
        
        # Return tokens or decoded text
        if tokenizer is not None:
            return tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated
