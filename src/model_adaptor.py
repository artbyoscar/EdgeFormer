"""
Adapter module to help EdgeFormer model integrate with the associative memory system.
"""

import torch
import logging

# Set up logging
logger = logging.getLogger('edgeformer.adapter')

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
        
        # Store the original forward method
        self._original_forward = model.forward
        
        # Associate memory with model
        model.memory = memory
        model.retriever = retriever
        model.using_memory = True
        
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
        outputs = self._original_forward(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states from the last layer
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
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
        outputs.logits = updated_logits
        
        # Update the hidden states list with the enhanced representation
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            all_hidden_states = list(outputs.hidden_states)
            all_hidden_states[-1] = integrated_hidden_states
            outputs.hidden_states = tuple(all_hidden_states)
        
        # Update the last hidden state if it exists in outputs
        if hasattr(outputs, 'last_hidden_state'):
            outputs.last_hidden_state = integrated_hidden_states
        
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
                    logits = outputs.logits
                    next_token_logits = logits[:, -1, :]
                    
                    # Get hidden states for the last token
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        last_hidden = outputs.hidden_states[-1][:, -1:, :].clone()
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
                    logits = outputs.logits
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
