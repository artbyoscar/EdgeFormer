# src/model/edgeformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import your config, or use the one defined above if this file is standalone for testing
try:
    from .config import EdgeFormerConfig
except ImportError:
    # This is for standalone testing if config.py is not in the same relative path
    # In your project, the above relative import should work.
    class EdgeFormerConfig: # Minimal fallback
        def __init__(self, vocab_size=32000, hidden_size=768, num_hidden_layers=12,
                     num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
                     hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
                     initializer_range=0.02, pad_token_id=0, **kwargs): # Added pad_token_id to minimal fallback
            self.vocab_size=vocab_size
            self.hidden_size=hidden_size
            self.num_hidden_layers=num_hidden_layers
            self.num_attention_heads=num_attention_heads
            self.intermediate_size=intermediate_size
            self.max_position_embeddings=max_position_embeddings
            self.hidden_dropout_prob=hidden_dropout_prob
            self.attention_probs_dropout_prob=attention_probs_dropout_prob
            self.layer_norm_eps=layer_norm_eps
            self.initializer_range=initializer_range
            self.pad_token_id=pad_token_id # Added here
            self.hidden_act = kwargs.get('hidden_act', "gelu") # Added for FeedForwardNetwork
            for key, value in kwargs.items(): setattr(self, key, value)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: EdgeFormerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_size)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len) for causal

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask # Apply mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.out_proj(context_layer)
        return attention_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: EdgeFormerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        
        if isinstance(config.hidden_act, str):
            if config.hidden_act == "gelu":
                 self.intermediate_act_fn = nn.GELU()
            elif config.hidden_act == "relu":
                 self.intermediate_act_fn = nn.ReLU()
            # Add other activations as needed
            else:
                 self.intermediate_act_fn = nn.GELU() # Default fallback
        elif callable(config.hidden_act): # If it's already a function or nn.Module
            self.intermediate_act_fn = config.hidden_act
        else:
            self.intermediate_act_fn = nn.GELU() # Default fallback
            
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, config: EdgeFormerConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.ffn = FeedForwardNetwork(config)
        
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout is often applied *before* the residual connection or *inside* attention/ffn,
        # but applying after the residual and before norm (or after norm) are also variants.
        # The original BERT applies dropout within attention and FFN, then to their outputs.
        # For simplicity here, applying to the output of sub-components before adding to residual.
        # self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        # self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention part
        attention_output = self.attention(hidden_states, attention_mask)
        # Residual connection + LayerNorm
        # Usually: hidden_states = norm(hidden_states + dropout(attention_output))
        hidden_states = self.norm1(hidden_states + attention_output) # Dropout is in attention.out_proj
        
        # Feed-Forward part
        ffn_output = self.ffn(hidden_states)
        # Residual connection + LayerNorm
        hidden_states = self.norm2(hidden_states + ffn_output) # Dropout is in ffn
        
        return hidden_states


class EdgeFormer(nn.Module):
    def __init__(self, config: EdgeFormerConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=getattr(config, 'pad_token_id', 0))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embeddings_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.output_projection_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _prepare_attention_mask(self, input_ids_shape, attention_mask, device):
        # input_ids_shape: (batch_size, seq_length)
        # attention_mask: (batch_size, seq_length)
        batch_size, seq_length = input_ids_shape
        
        # Causal mask for autoregressive models (GPT-like)
        # For BERT-like models, this part creating the causal mask would be omitted or different.
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool, device=device)).view(1, 1, seq_length, seq_length)
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device) # Attend to all tokens
        
        # Convert attention_mask to a bias: 0 for attended, -10000 for masked
        # Expected shape for additive attention bias: (batch_size, num_heads, seq_length, seq_length)
        # or (batch_size, 1, seq_length, seq_length) for broadcasting
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Combine with causal mask. This setup assumes a decoder-style model.
        # If your model can be an encoder, you'd need conditional logic here.
        final_mask = extended_attention_mask
        # Add causal mask for decoder behavior (e.g., GPT)
        # This specific addition will make padding tokens also "causally" masked from future tokens
        # which might not be standard. Standard Hugging Face GPT2 combines masks differently.
        # A common way for causal + padding:
        # 1. Start with causal mask (ones on lower triangle, zeros on upper)
        # 2. Create padding mask (ones where input is not padding, zeros where it is)
        # 3. Combine: e.g. causal_mask * padding_mask.unsqueeze(1).unsqueeze(2)
        # Then convert 0s to large negative for softmax.
        # For simplicity in this example, we're making it causal and applying padding.
        
        # The provided `_prepare_attention_mask` had a potentially complex interaction.
        # Let's simplify for a standard causal decoder with padding support.
        # seq_len_to = input_ids_shape[-1]
        # causal_mask = torch.full((seq_length, seq_len_to), -10000.0, device=device)
        # causal_mask = torch.triu(causal_mask, diagonal=1) # Upper triangle is masked
        # causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_len_to)
        
        # # Combine with provided attention_mask (padding mask)
        # if extended_attention_mask is not None:
        #      final_mask = causal_mask + extended_attention_mask # Element-wise adding masks
        # else:
        #      final_mask = causal_mask
        # This is a standard approach to create a causal mask.
        _causal_mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
        _causal_mask = torch.triu(_causal_mask, diagonal=1) # True for upper triangle (masked)
        
        final_mask = torch.zeros(batch_size, 1, seq_length, seq_length, dtype=torch.float32, device=device)
        final_mask.masked_fill_(_causal_mask[None, None, :, :], -10000.0)

        # Apply padding mask from attention_mask
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)[:, None, None, :] # True where padded
            final_mask.masked_fill_(padding_mask, -10000.0)

        return final_mask

    def forward(self, input_ids, attention_mask=None, past_key_values=None, output_attentions=False, output_hidden_states=False, use_cache=False):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0) # Shape (1, seq_len)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids) # Position IDs will broadcast
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = self.embeddings_dropout(hidden_states)

        collected_hidden_states = () if output_hidden_states else None
        collected_attentions = () if output_attentions else None
        
        processed_attention_mask = self._prepare_attention_mask(input_ids.shape, attention_mask, device)

        for i, block in enumerate(self.transformer_blocks):
            if output_hidden_states:
                collected_hidden_states += (hidden_states,)
            
            # TODO: Integrate past_key_values and use_cache for generation efficiency
            # TODO: Integrate different attention_type logic (GQA, MLA, etc.)
            # TODO: Integrate specific EdgeFormer innovations here
            # For now, block only returns hidden_states. If it returned attentions:
            # layer_output_tuple = block(hidden_states, attention_mask=processed_attention_mask)
            # hidden_states = layer_output_tuple[0]
            # if output_attentions:
            #     collected_attentions += (layer_output_tuple[1],)
            hidden_states = block(hidden_states, attention_mask=processed_attention_mask)


        hidden_states = self.output_projection_layer_norm(hidden_states)

        if output_hidden_states:
            collected_hidden_states += (hidden_states,)

        logits = self.lm_head(hidden_states)

        # For Hugging Face compatibility, the output is often a custom ModelOutput object
        # For simplicity, returning a dictionary
        return {
            "logits": logits,
            "past_key_values": past_key_values, # Placeholder for now
            "hidden_states": collected_hidden_states,
            "attentions": collected_attentions, # Placeholder for now
        }

    @staticmethod
    def from_pretrained(model_name_or_path, config: EdgeFormerConfig = None, **kwargs):
        if config is None:
            print(f"Warning: No config provided to from_pretrained for {model_name_or_path}. Using default EdgeFormerConfig based on kwargs or defaults.")
            # Attempt to filter kwargs that are valid for EdgeFormerConfig
            # This is a basic way; a more robust way would inspect EdgeFormerConfig.__init__ signature
            valid_config_keys = EdgeFormerConfig().__dict__.keys()
            config_params = {k: v for k, v in kwargs.items() if k in valid_config_keys}
            config = EdgeFormerConfig(**config_params)
        
        model = EdgeFormer(config)
        print(f"EdgeFormer model '{model_name_or_path}' initialized (Note: weights are NOT loaded from path in this placeholder).")
        # Add actual weight loading logic here, e.g.:
        # try:
        #     state_dict = torch.load(Path(model_name_or_path) / "pytorch_model.bin", map_location="cpu")
        #     model.load_state_dict(state_dict)
        # except Exception as e:
        #     print(f"Could not load weights from {model_name_or_path}: {e}")
        return model

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, attention_mask=None, 
                 eos_token_id=None, pad_token_id=None, **kwargs): # Added eos_token_id
        self.eval()
        
        if eos_token_id is None and hasattr(self.config, 'eos_token_id'):
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None and hasattr(self.config, 'pad_token_id'):
            pad_token_id = self.config.pad_token_id


        for _ in range(max_new_tokens):
            # If attention_mask is not provided, create a default one
            current_attention_mask = attention_mask
            if current_attention_mask is None:
                current_attention_mask = torch.ones_like(input_ids)

            model_inputs = {"input_ids": input_ids, "attention_mask": current_attention_mask}
            
            outputs = self(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # Update attention_mask for the new token
            if attention_mask is not None: # Only if user provided an initial mask
                attention_mask = F.pad(attention_mask, (0, 1), mode='constant', value=1) # Pad with 1 (attended)

            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break
        
        return input_ids