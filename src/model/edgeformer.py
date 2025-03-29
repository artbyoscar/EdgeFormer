import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from src.utils.kv_cache_manager import KVCacheManager

from .config import EdgeFormerConfig
from .transformer_block import EdgeFormerBlock
from ..utils.long_sequence import process_long_document
from src.utils.htps_budget_manager import HTPSBudgetManager


# ---- Embeddings Implementation ----
class EdgeFormerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Initialize position ids
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize kv_cache_manager attribute
        self.kv_cache_manager = None
        self.config = config  # Also need to store config for later use

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        # Initialize KV Cache Manager if needed
        if self.kv_cache_manager is None:
            self.kv_cache_manager = KVCacheManager(
                max_batch_size=1,  # Adjust based on batch size
                max_seq_length=self.config.max_position_embeddings,
                num_layers=self.config.num_hidden_layers,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                device=input_ids.device
            )
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        seq_length = inputs_embeds.size(1)

        # Add sequence length check and warning
        max_allowed_length = self.position_embeddings.weight.size(0)
        if seq_length > max_allowed_length:
            print(f"WARNING: Input sequence length ({seq_length}) exceeds maximum position embeddings ({max_allowed_length})")
            # Option 1: Truncate sequence with warning
            # seq_length = max_allowed_length
            # inputs_embeds = inputs_embeds[:, :max_allowed_length, :]

            # Option 2: Extend position embeddings dynamically - use with caution
            # This creates extended position embeddings for this forward pass only
            extended_position_embeddings = nn.Embedding(seq_length, self.position_embeddings.weight.size(1))
            extended_position_embeddings.to(self.position_embeddings.weight.device)
            # Copy existing weights
            with torch.no_grad():
                extended_position_embeddings.weight[:max_allowed_length] = self.position_embeddings.weight
                # Initialize new positions with pattern continuation
                for i in range(max_allowed_length, seq_length):
                    offset = i % max_allowed_length
                    extended_position_embeddings.weight[i] = self.position_embeddings.weight[offset]

            # Use extended embeddings for this forward pass
            position_ids = torch.arange(seq_length, device=inputs_embeds.device).expand((1, -1))
            position_embeds = extended_position_embeddings(position_ids)
        else:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            position_embeds = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def memory_aware_forward(self, input_ids, max_memory_mb=1000):
        """Automatically adjust processing based on available memory.
        Args:
            input_ids: Input token IDs
            max_memory_mb: Maximum memory to use in MB
        
         Returns:
            Model outputs
        """
        # Estimate memory needs based on sequence length
        seq_length = input_ids.size(1)
        estimated_memory = seq_length * self.config.hidden_size * 4 / (1024 * 1024)  # Rough estimate in MB

        if estimated_memory > max_memory_mb:
            # Switch to chunked processing
            print(f"Sequence too long for available memory ({estimated_memory:.2f} MB), using chunked processing")
            return process_long_document(self, input_ids)
        else:
            # Use standard processing
            return self.forward(input_ids)
        
    def process_in_chunks(self, input_ids, chunk_size=4096, overlap=512):
        """Process input in chunks for long sequences.

        Args:
            input_ids: Input token IDs
            chunk_size: Maximum sequence length to process at once
            overlap: Number of tokens to overlap between chunks

        Returns:
            Model outputs
        """

        return process_long_document(self, input_ids, chunk_size, overlap)




# ---- LM Head Implementation ----
class EdgeFormerLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    @property
    def device(self):
        """Return the device where the model parameters are stored."""
        return next(self.parameters()).device
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.decoder(hidden_states) + self.bias
        return logits

# ---- SimpleTokenizer----
class SimpleTokenizer:
    """Simple tokenizer class for when we don't have a full tokenizer object."""
    
    def __init__(self, char_to_idx=None, idx_to_char=None, vocab_size=50000):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        
    def encode(self, text, return_tensors=None):
        """Simple encoding function."""
        if self.char_to_idx:
            tokens = [self.char_to_idx.get(c, 0) for c in text]
        else:
            tokens = [ord(c) % self.vocab_size for c in text]
            
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens
        
    def decode(self, token_ids):
        """Simple decoding function."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
            
        text = ""
        for token_id in token_ids:
            if self.idx_to_char:
                text += self.idx_to_char.get(token_id, '')
            else:
                text += chr(token_id % 128)
        return text
    
    def __len__(self):
        """Return the vocabulary size."""
        return self.vocab_size

# ---- Main EdgeFormer Model ----
class EdgeFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kv_cache_manager = None
        
        # Embeddings
        self.embeddings = EdgeFormerEmbeddings(config)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            EdgeFormerBlock(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # LM head
        self.lm_head = EdgeFormerLMHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation tasks."""
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # Only take the last token of the input for continued generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        
        if attention_mask is not None and position_ids is None:
            # Create position_ids from input_ids and attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
    def estimate_confidence(self, logits: torch.Tensor) -> float:
        """
        Estimate model confidence from output logits.
    
        Args:
            logits: Output logits from the model
        
        Returns:
            float: Confidence score between 0 and 1
        """
        # Get probabilities from logits
        probs = torch.softmax(logits, dim=-1)
    
        # Get maximum probability as confidence
        max_probs = torch.max(probs, dim=-1)[0]
    
        # Average confidence across the sequence
        avg_confidence = max_probs.mean().item()
    
        return avg_confidence   
     
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sliding_window_size=None,
    ):
        
        # Add sequence length validation
        if input_ids is not None:
            seq_length = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.size(1)
        else:
            raise ValueError("You must provide either input_ids or inputs_embeds")
        
        # Log sequence length for debugging
        print(f"Processing sequence of length: {seq_length}")

        # Add memory tracking
        try:
            # Only track if running with the debug flag
            if os.environ.get('EDGEFORMER_DEBUG') == '1':
                print(f"Input shape: {input_ids.shape if input_ids is not None else inputs_embeds.shape}")

                # Record initial memory
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated() / 1024 / 1024
                    print(f"Starting memory: {start_mem:.2f} MB")

                # Track memory after key operations
                def log_memory(step):
                    if torch.cuda.is_available():
                        current_mem = torch.cuda.memory_allocated() / 1024 / 1024
                        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                        print(f"{step}: Current memory: {current_mem:.2f} MB, Peak: {peak_mem:.2f} MB")
        except ImportError:
            # Continue without tracking if torch is not available
            pass

        max_supported_length = self.config.max_position_embeddings
        if seq_length > max_supported_length:
            print(f"WARNING: Sequence length {seq_length} exceeds model's configured maximum {max_supported_length}")
            # Choose handling strategy: truncate, error, or extend (as implemented above)

        # Set defaults
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, "use_cache") else False
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        # Force garbage collection before getting embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(
                input_ids=input_ids, 
                position_ids=position_ids
            )

        # Add memory tracking call
        if os.environ.get('EDGEFORMER_DEBUG') == '1' and torch.cuda.is_available():
            log_memory("After embeddings")
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Create causal mask if needed
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
        
        # We can provide a self-attention mask of dimensions [batch_size, 1, seq_length, seq_length]
        # that is causal and consistent with past_key_values
        extended_attention_mask = self._prepare_attention_mask(
            attention_mask, (batch_size, seq_length), past_key_values
        )
        
        # Initialize outputs
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None

        # Force garbage collection before processing layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=None,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                sliding_window_size=sliding_window_size,
            )
            
            hidden_states = layer_outputs[0]

            # Add memory tracking call
            if os.environ.get('EDGEFORMER_DEBUG') == '1' and torch.cuda.is_available():
                log_memory(f"After layer {i}")
            
            if use_cache:
                all_past_key_values = all_past_key_values + (layer_outputs[1],)
                
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Add memory tracking call
        if os.environ.get('EDGEFORMER_DEBUG') == '1' and torch.cuda.is_available():
            log_memory("After final layer norm")
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Force garbage collection before LM head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate logits
        logits = self.lm_head(hidden_states)

        # Add memory tracking call
        if os.environ.get('EDGEFORMER_DEBUG') == '1' and torch.cuda.is_available():
            log_memory("After LM head")
        
        # Prepare output dictionary
        outputs = {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "past_key_values": all_past_key_values,
        }

        # Add loss calculation if labels are provided
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            outputs["loss"] = loss

        return outputs
    
    def _prepare_attention_mask(self, attention_mask, input_shape, past_key_values=None):
        """Prepare attention mask for processing."""
        # Get batch size and sequence length
        batch_size, seq_length = input_shape

        # Add logging for debugging
        print(f"Preparing attention mask: batch_size={batch_size}, seq_length={seq_length}")
        if past_key_values is not None:
            print(f"Past key values present with shape: {past_key_values[0][0].shape if past_key_values[0][0] is not None else 'None'}")

        # Calculate past length from past_key_values
        past_length = 0
        if past_key_values is not None and isinstance(past_key_values, tuple) and len(past_key_values) > 0:
            if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
                if past_key_values[0][0] is not None:
                    past_length = past_key_values[0][0].size(1)

        # Total sequence length including past
        total_length = past_length + seq_length
        print(f"Total attention length: {total_length} (past: {past_length}, current: {seq_length})")

        # If attention_mask is already 4D, assume it's already prepared
        if attention_mask.dim() == 4:
            return attention_mask

        # For 2D attention mask, create a new mask of the correct total length
        if attention_mask.dim() == 2:
            # Make sure mask shape is compatible with sequence length
            if attention_mask.size(1) != seq_length:
                # If sizes don't match, we have two options:
                # 1. For shorter masks (common with continuation), just use all ones
                # 2. For longer masks (shouldn't happen normally), truncate
                if attention_mask.size(1) < seq_length:
                    # This should not happen normally, but handle it just in case
                    attention_mask = torch.ones((batch_size, seq_length), device=attention_mask.device)
                else:
                    # Truncate longer mask to match seq_length
                    attention_mask = attention_mask[:, :seq_length]

            # Create extended mask including past tokens if needed
            if past_length > 0:
                # Create new mask of total length
                extended_attention_mask = torch.ones(
                    (batch_size, total_length), device=attention_mask.device
                )
                # Put original attention mask in the right position
                extended_attention_mask[:, past_length:] = attention_mask
            else:
                extended_attention_mask = attention_mask

            # Add dimension to convert to 3D [batch_size, 1, seq_length]
            extended_attention_mask = extended_attention_mask.unsqueeze(1)
        else:
            # For 3D or other dim masks, just use as is
            extended_attention_mask = attention_mask

        # Add another dimension if needed to make 4D
        if extended_attention_mask.dim() == 3:
            extended_attention_mask = extended_attention_mask.unsqueeze(1)

        # Convert attention mask values:
        # 0.0 for masked positions (don't attend)
        # 1.0 for positions we can attend to
        # Then convert to attention scores: (1.0 - mask) * -10000.0
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Apply causal mask (only attend to tokens up to current position)
        if extended_attention_mask.size(-1) == extended_attention_mask.size(-2):
            # Only apply causal mask for square attention masks
            causal_mask = torch.ones(
                (extended_attention_mask.size(-1), extended_attention_mask.size(-1)),
                dtype=torch.bool, 
                device=attention_mask.device
            )
            # Fill upper triangular (future tokens)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            # Reshape for broadcasting
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            # Apply causal mask
            extended_attention_mask = extended_attention_mask.masked_fill(causal_mask, -10000.0)

        # Expand mask for all heads if needed
        if extended_attention_mask.size(1) == 1:
            # Get number of attention heads from config
            if hasattr(self, 'config') and hasattr(self.config, 'num_attention_heads'):
                num_heads = self.config.num_attention_heads
            else:
                num_heads = 1  # Default if not available
            
            # Expand for all heads
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, 
                num_heads, 
                extended_attention_mask.size(2), 
                extended_attention_mask.size(3)
            )

        return extended_attention_mask

    def forward_with_hidden_states(self, input_ids, attention_mask=None):
        """Forward pass that returns both logits and hidden states for recurrent processing.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
    
        Returns:
            tuple: (logits, hidden_states)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
    
        # Get embeddings
        embeddings = self.embeddings(input_ids)

        # Initialize hidden states list to store intermediate values
        all_hidden_states = [embeddings]

        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:  # Changed from self.encoder.layer to self.layers
            # Use the first output (hidden states) from the layer
            hidden_states = layer(
                hidden_states,
                attention_mask=self._prepare_attention_mask(attention_mask, input_ids.shape)
            )[0]
            all_hidden_states.append(hidden_states)

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        all_hidden_states.append(hidden_states)

        # Get logits from final hidden states
        logits = self.lm_head(hidden_states)

        return logits, all_hidden_states
  
    def generate(
        self,
        input_text,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=None,
        eos_token_id=None,
        budget_manager=None,
        task_complexity=None,
    ):
        """Generate text using the model with optional budget management.

        Args:
            input_text: Either a string or tokenized input_ids tensor
            attention_mask: Optional attention mask
            max_length: Maximum length of the generated sequence
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to return
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            budget_manager: Optional HTPSBudgetManager for controlling compute
            task_complexity: Optional task complexity score for budget management
    
        Returns:
            Generated text
        """
        self.eval()
        device = next(self.parameters()).device

        # Initialize budget manager if specified
        using_budget = budget_manager is not None
        if using_budget:
            budget_manager.reset()

        # Handle string input by tokenizing it
        if isinstance(input_text, str):
            # If we're using character-level tokenization, convert string to tokens
            if hasattr(self, 'char_to_idx'):
                tokens = [self.char_to_idx.get(c, 0) for c in input_text]
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            else:
                # Fallback to simple character encoding
                tokens = [ord(c) % self.config.vocab_size for c in input_text]
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        else:
            # Use provided tensor
            input_ids = input_text
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension

        # Store the prompt for later reconstruction
        prompt_length = input_ids.shape[1]

        # Set default token IDs
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.config, 'pad_token_id', 0)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.config, 'eos_token_id', None)

        batch_size = input_ids.shape[0]

        # Initialize past key values for efficient generation
        past_key_values = None

        # Keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(batch_size).fill_(1)

        # Clone input_ids to avoid modifying the original
        input_ids = input_ids.clone()

        # Generate until we reach max_length or all sequences are finished
        cur_len = input_ids.shape[1]
        while cur_len < max_length and unfinished_sequences.sum().item() > 0:
            # Prepare inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values, attention_mask=attention_mask
            )
    
            # Forward pass
            model_inputs['use_cache'] = True
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]
        
            # Apply budget enforcement if using budget manager
            if using_budget:
                # Check if we need to extend thinking or enforce budget
                if hasattr(self, 'tokenizer'):
                    tokenizer = self.tokenizer
                else:
                    # Create simple tokenizer with char mappings if available
                    tokenizer = SimpleTokenizer(
                        char_to_idx=getattr(self, 'char_to_idx', None),
                        idx_to_char=getattr(self, 'idx_to_char', None),
                        vocab_size=self.config.vocab_size
                    )
                
                input_ids, continue_gen = budget_manager.enforce_budget(
                    tokenizer, input_ids, next_token_logits, task_complexity
                )
            
                # Stop generation if budget is exhausted
                if not continue_gen:
                    break
                
                # If input_ids were extended with thinking tokens, update past key values
                if past_key_values is not None and input_ids.shape[1] > cur_len:
                    # Need to recompute for the added tokens
                    extension_inputs = self.prepare_inputs_for_generation(
                        input_ids[:, cur_len:], past_key_values=past_key_values, attention_mask=attention_mask
                    )
                    extension_inputs['use_cache'] = True
                    extension_outputs = self.forward(**extension_inputs)
                    past_key_values = extension_outputs.get("past_key_values", None)
                    # Update next token logits after extension
                    next_token_logits = extension_outputs["logits"][:, -1, :]
                    # Update current length
                    cur_len = input_ids.shape[1]
                    continue
    
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
    
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
    
            # Select next tokens
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    k = min(top_k, next_token_logits.size(-1))  # Make sure k isn't larger than vocab size
                    values, _ = torch.topk(next_token_logits, k)
                    min_value = values[..., -1, None]
                    indices_to_remove = next_token_logits < min_value
                    next_token_logits[indices_to_remove] = -float("inf")
        
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
            
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")
        
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
        
            # Update token count in budget manager if using
            if using_budget:
                budget_manager.update_token_count(1)
    
            # Update unfinished sequences
            if eos_token_id is not None:
                # Set EOS token probability to 0 for unfinished sequences
                tokens_to_add = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                # Mark sequences with EOS token as finished
                eos_in_next_tokens = next_tokens == eos_token_id
                unfinished_sequences = unfinished_sequences * ~eos_in_next_tokens
            else:
                tokens_to_add = next_tokens
    
            # Add tokens to sequences
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
    
            # Update attention mask if needed
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))],
                    dim=-1
                )
    
            # Update progress
            cur_len += 1
    
            # Update past key values
            past_key_values = outputs.get("past_key_values", None)

        # Convert back to text if input was string
        if isinstance(input_text, str):
            generated_text = ""
            for token_idx in input_ids[0]:
                if hasattr(self, 'idx_to_char'):
                    # If we have a character mapping, use it
                    generated_text += self.idx_to_char.get(token_idx.item(), '')
                else:
                    # Fallback to simple ASCII
                    generated_text += chr(token_idx.item() % 128)
            return generated_text
        else:
            # Return tensor if input was tensor
            return input_ids
    
    def continue_generation(self, new_tokens, past_key_values):
        """
        Helper method to continue generation with KV cache.
    
        Args:
            new_tokens: New token IDs to generate from (tensor of shape [batch_size, new_seq_length])
            past_key_values: Past key-values from previous generation step (can be a string ID or tensor tuple)
        
        Returns:
            Dictionary with generated outputs
        """
        # Get past length from past_key_values
        past_length = 0
        if past_key_values is not None and isinstance(past_key_values, tuple) and len(past_key_values) > 0:
            if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
                if past_key_values[0][0] is not None:
                    past_length = past_key_values[0][0].size(1)
    
        # Get batch size and new sequence length
        batch_size = new_tokens.shape[0]
        new_seq_length = new_tokens.shape[1]
    
        # Create attention mask that spans both past tokens and new tokens
        total_seq_length = past_length + new_seq_length
        attention_mask = torch.ones((batch_size, total_seq_length), device=new_tokens.device)
    
        # Run forward pass with prepared inputs
        outputs = self(
            input_ids=new_tokens,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
    
        return outputs