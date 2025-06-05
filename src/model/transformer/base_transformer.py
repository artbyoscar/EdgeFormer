"""Base Transformer implementation for EdgeFormer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import EdgeFormerConfig
from .mla import MultiHeadLatentAttention
from .gqa import GroupedQueryAttention

class EdgeFormerEmbeddings(nn.Module):
    """Embeddings for the EdgeFormer model."""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids, position_ids=None):
        """Forward pass."""
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class EdgeFormerSelfAttention(nn.Module):
    """Self-attention layer with support for different attention types."""
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
    
        self.attention_type = config.attention_type
    
        if self.attention_type == "mla":
            # Use Multi-Head Latent Attention
            self.mla = MultiHeadLatentAttention(config)
        elif self.attention_type == "gqa":
            # Use Grouped-Query Attention
            self.gqa = GroupedQueryAttention(config)
        else:
            # Standard attention
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
    
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.sliding_window_size = getattr(config, "sliding_window_size", 512)
        
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """Forward pass with different attention types."""
        if self.attention_type == "mla":
            return self.mla(
                hidden_states,
                attention_mask,
                head_mask,
                past_key_value,
                output_attentions,
            )
        elif self.attention_type == "gqa":
            return self.gqa(
                hidden_states,
                attention_mask,
                head_mask,
                past_key_value,
                output_attentions,
            )
            
        mixed_query_layer = self.query(hidden_states)
        
        # If past key value is used, only the last token needs to be processed
        if past_key_value is not None:
            mixed_query_layer = mixed_query_layer[:, -1:, :]
            past_key, past_value = past_key_value
            key_layer = past_key
            value_layer = past_value
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # Take the dot product to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply sliding window attention if specified
        if self.attention_type == "sliding_window" and attention_mask is not None:
            # Create a sliding window mask
            seq_length = hidden_states.size(1)
            window_mask = torch.ones(
                (seq_length, seq_length), device=attention_scores.device
            )
            
            for i in range(seq_length):
                window_start = max(0, i - self.sliding_window_size // 2)
                window_end = min(seq_length, i + self.sliding_window_size // 2 + 1)
                window_mask[i, window_start:window_end] = 0
                
            # Add large negative values to positions outside window
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores - 10000.0 * window_mask
        
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        if past_key_value is None:
            past_key_value = (key_layer, value_layer)
            
        outputs = outputs + (past_key_value,)
        
        return outputs

class EdgeFormerSelfOutput(nn.Module):
    """Output layer for self-attention."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, input_tensor):
        """Forward pass."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class EdgeFormerAttention(nn.Module):
    """Attention module combining self-attention and output layers."""
    
    def __init__(self, config):
        super().__init__()
        self.self = EdgeFormerSelfAttention(config)
        self.output = EdgeFormerSelfOutput(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """Forward pass."""
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class EdgeFormerIntermediate(nn.Module):
    """Intermediate layer for the Transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = F.gelu
        elif config.hidden_act == "relu":
            self.intermediate_act_fn = F.relu
        else:
            self.intermediate_act_fn = F.gelu
            
    def forward(self, hidden_states):
        """Forward pass."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class EdgeFormerOutput(nn.Module):
    """Output layer for the Transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, input_tensor):
        """Forward pass."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class EdgeFormerLayer(nn.Module):
    """Transformer layer combining attention and feed-forward network."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = EdgeFormerAttention(config)
        self.intermediate = EdgeFormerIntermediate(config)
        self.output = EdgeFormerOutput(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """Forward pass."""
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:]  # add attentions if we output them
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + outputs
        
        return outputs

class EdgeFormerEncoder(nn.Module):
    """Transformer encoder with multiple layers."""
    
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EdgeFormerLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """Forward pass through all layers."""
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                past_key_value,
                output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None
        )

class EdgeFormerModel(nn.Module):
    """Base EdgeFormer model combining all components."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = EdgeFormerEmbeddings(config)
        self.encoder = EdgeFormerEncoder(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """Forward pass."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Prepare attention mask for the attention layers
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers
            
        embedding_output = self.embeddings(input_ids, position_ids)
        
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask,
            past_key_values,
            output_attentions,
            output_hidden_states,
        )
        
        sequence_output = encoder_outputs[0]
        
        return sequence_output, encoder_outputs[1:]

class EdgeFormerLMHead(nn.Module):
    """Language modeling head for EdgeFormer."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, features):
        """Forward pass."""
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        
        return x

class EdgeFormer(nn.Module):
    """Complete EdgeFormer model with LM head."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = EdgeFormerModel(config)
        self.lm_head = EdgeFormerLMHead(config)
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """Forward pass."""
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            head_mask,
            past_key_values,
            output_attentions,
            output_hidden_states,
        )
        
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits, transformer_outputs[1:]
    
    def generate(
        self,
        input_ids,
        max_length=None,
        min_length=None,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        pad_token_id=None,
        eos_token_id=None,
        attention_mask=None,
    ):
        """Generate text using the model."""
        # Set default values
        max_length = max_length if max_length is not None else 20
        min_length = min_length if min_length is not None else 0
        do_sample = do_sample if do_sample is not None else False
        temperature = temperature if temperature is not None else 1.0
        top_k = top_k if top_k is not None else 50
        top_p = top_p if top_p is not None else 1.0
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.0
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        # Initialize generation variables
        batch_size = input_ids.shape[0]
        generated = input_ids
        past = None
        
        # Generate tokens
        for _ in range(max_length - input_ids.shape[1]):
            if past is None:
                outputs = self(generated, attention_mask=attention_mask)
            else:
                outputs = self(generated[:, -1:], past_key_values=past)
                
            logits = outputs[0][:, -1, :]
            past = outputs[1]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated[i]:
                        if previous_token.item() != pad_token_id:
                            logits[i, previous_token.item()] /= repetition_penalty
            
            # If min_length is not reached, set eos token prob to 0
            if min_length > 0 and generated.shape[1] < min_length:
                logits[:, eos_token_id] = -float("inf")
            
            # Sample or greedy decode
            if do_sample:
                # Apply top_k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float("inf")
                
                # Apply top_p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        logits[batch_idx, indices_to_remove] = -float("inf")
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decode
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            # Append next token to generated
            generated = torch.cat((generated, next_token), dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                )
            
            # Check if EOS token was generated
            if next_token[0, 0].item() == eos_token_id:
                break
        
        return generated
