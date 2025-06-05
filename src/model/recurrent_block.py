import torch
import torch.nn as nn
import math

class RecurrentTransformerBlock(nn.Module):
    """
    A recurrent transformer block that can be iterated multiple times at inference
    to effectively increase model depth without additional parameters.
    """
    def __init__(self, config):
        """
        Initialize the recurrent transformer block.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Iteration count tracking
        self.current_iteration = 0
        self.max_iterations = 1
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass through the recurrent transformer block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Apply attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Convert to format expected by nn.MultiheadAttention
        # [seq_len, batch_size, hidden_size]
        hidden_states_T = hidden_states.transpose(0, 1)
        
        # Create attention mask in the format expected by nn.MultiheadAttention
        if attention_mask is not None:
            # Convert boolean mask to float
            attn_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float("-inf")
            ).masked_fill(attention_mask == 1, float(0.0))
            
            # Create causal mask (lower triangular)
            seq_len = hidden_states.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            
            # Combine masks
            if attn_mask.dim() == 2:
                # [batch_size, seq_len] -> [seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(1) + causal_mask
            else:
                # [batch_size, seq_len, seq_len] -> [seq_len, seq_len]
                attn_mask = attn_mask + causal_mask
        else:
            # Just use causal mask
            seq_len = hidden_states.size(1)
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
        
        # Self-attention
        attn_output, _ = self.attention(
            hidden_states_T, hidden_states_T, hidden_states_T,
            attn_mask=attn_mask
        )
        
        # Transpose back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(0, 1)
        
        # Add residual connection
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.dropout(self.ffn(hidden_states))
        
        # Track iteration
        self.current_iteration += 1
        
        return hidden_states
    
    def reset_iteration_count(self):
        """Reset the iteration counter"""
        self.current_iteration = 0
    
    def set_max_iterations(self, max_iterations):
        """Set the maximum number of iterations"""
        self.max_iterations = max_iterations


class ValueEstimator(nn.Module):
    """
    Estimates the value (quality) of the current hidden state to determine
    when to stop iterating the recurrent transformer block.
    """
    def __init__(self, hidden_size):
        """
        Initialize the value estimator.
        
        Args:
            hidden_size: Hidden size of the model
        """
        super().__init__()
        
        # Value estimation network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Convergence threshold
        self.convergence_threshold = 0.01
        
        # Track previous values for convergence detection
        self.prev_values = None
    
    def forward(self, hidden_states):
        """
        Estimate the value of the current hidden state.
        
        Args:
            hidden_states: Hidden state tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            value: Estimated value tensor of shape [batch_size, seq_len, 1]
        """
        # Compute value for each position
        value = self.value_net(hidden_states)
        
        # Return value estimate
        return value
    
    def check_convergence(self, hidden_states):
        """
        Check if the hidden states have converged (stopped changing significantly).
        
        Args:
            hidden_states: Hidden state tensor
            
        Returns:
            bool: True if converged, False otherwise
        """
        value = self.forward(hidden_states)
        
        if self.prev_values is None:
            self.prev_values = value
            return False
        
        # Compute change in value
        value_change = torch.abs(value - self.prev_values).mean().item()
        self.prev_values = value
        
        # Check convergence
        return value_change < self.convergence_threshold
    
    def reset(self):
        """Reset the convergence tracking"""
        self.prev_values = None


class RecurrentDepthProcessor:
    """
    Manages the recurrent depth processing, controlling the number of iterations
    and detecting convergence.
    """
    def __init__(self, recurrent_block, value_estimator):
        """
        Initialize the recurrent depth processor.
        
        Args:
            recurrent_block: RecurrentTransformerBlock instance
            value_estimator: ValueEstimator instance
        """
        self.recurrent_block = recurrent_block
        self.value_estimator = value_estimator
        
        # Configuration
        self.min_iterations = 1
        self.max_iterations = 32
        self.check_convergence = True
    
    def process(self, hidden_states, attention_mask=None):
        """
        Process the hidden states through multiple iterations of the recurrent block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            
        Returns:
            Output tensor after multiple iterations
        """
        # Reset iteration count and value estimator
        self.recurrent_block.reset_iteration_count()
        self.value_estimator.reset()
        
        # Iterate the recurrent block
        for i in range(self.max_iterations):
            # Process hidden states
            hidden_states = self.recurrent_block(hidden_states, attention_mask)
            
            # Check if we've done enough iterations
            if i >= self.min_iterations - 1:
                # Check for convergence
                if self.check_convergence and self.value_estimator.check_convergence(hidden_states):
                    print(f"Converged after {i+1} iterations")
                    break
            
            # Optional: print value estimate
            if i % 5 == 0:
                value = self.value_estimator(hidden_states).mean().item()
                print(f"Iteration {i+1}, Value: {value:.4f}")
        
        return hidden_states
