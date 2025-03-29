"""
Adapter module to help EdgeFormer model integrate with the associative memory system.
"""

import torch

class EdgeFormerMemoryAdapter:
    """
    Helper class to integrate EdgeFormer model with associative memory components.
    """
    
    def __init__(self, model, memory, retriever):
        """
        Initialize the adapter with model and memory components.
        
        Args:
            model: EdgeFormer model instance
            memory: HTPSMemory instance
            retriever: AttentionBasedRetriever instance
        """
        self.model = model
        self.memory = memory
        self.retriever = retriever
        
        # Store the original forward method
        self._original_forward = model.forward
        
        # Associate memory with model
        model.memory = memory
        model.retriever = retriever
        model.using_memory = True
        
        # Add method to model for memory integration
        model.associate_memory = self.associate_memory
        
        # Override the model's forward method
        model.forward = self._forward_with_memory
        
        # Add attribute to track recurrent iterations
        model.recurrent_iterations = 0
    
    def associate_memory(self, memory, retriever):
        """
        Associate memory components with the model.
        
        Args:
            memory: HTPSMemory instance
            retriever: AttentionBasedRetriever instance
        """
        self.memory = memory
        self.retriever = retriever
        self.model.memory = memory
        self.model.retriever = retriever
        self.model.using_memory = True
    
    def _forward_with_memory(self, input_ids=None, attention_mask=None, token_type_ids=None,
                           position_ids=None, head_mask=None, inputs_embeds=None,
                           output_attentions=None, output_hidden_states=None, return_dict=None,
                           past_key_values=None, use_cache=None, labels=None, **kwargs):
        """
        Forward pass with memory integration.
        """
        # Call the original forward method
        outputs = self._original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always need hidden states for memory
            return_dict=True,  # Always return dict for easier access
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            **kwargs
        )
        
        # Check if we should use memory
        use_memory = kwargs.get('use_memory', True) and hasattr(self.model, 'using_memory') and self.model.using_memory
        
        if use_memory:
            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]
            
            # Check if we have retrieved memory embeddings from a previous step
            retrieved_embeddings = kwargs.get('retrieved_memory_embeddings', None)
            retrieved_scores = kwargs.get('retrieved_memory_scores', None)
            
            # Apply memory integration if we have retrieved embeddings
            if retrieved_embeddings is not None and retrieved_scores is not None:
                # Move tensors to the same device as the model
                if isinstance(retrieved_embeddings, torch.Tensor):
                    retrieved_embeddings = retrieved_embeddings.to(hidden_states.device)
                if isinstance(retrieved_scores, torch.Tensor):
                    retrieved_scores = retrieved_scores.to(hidden_states.device)
                
                # Add batch dimension if necessary
                if len(retrieved_embeddings.shape) == 2:  # [num_memories, hidden_size]
                    retrieved_embeddings = retrieved_embeddings.unsqueeze(0)
                if len(retrieved_scores.shape) == 1:  # [num_memories]
                    retrieved_scores = retrieved_scores.unsqueeze(0)
                
                # Apply memory retriever to enhance hidden states
                memory_enhanced_hidden = self.retriever(
                    hidden_states, retrieved_embeddings, retrieved_scores
                )
                
                # Create a new outputs object with the enhanced hidden states
                outputs.last_hidden_state = memory_enhanced_hidden
                
                # Update the hidden states list with the enhanced representation
                all_hidden_states = list(outputs.hidden_states)
                all_hidden_states[-1] = memory_enhanced_hidden
                outputs.hidden_states = tuple(all_hidden_states)
        
        return outputs