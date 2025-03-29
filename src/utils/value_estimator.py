import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedValueEstimator(nn.Module):
    """
    Enhanced value estimator with pattern recognition capabilities for recurrent depth processing.
    """
    def __init__(self, hidden_size, config=None):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention-weighted pooling
        self.attention_query = nn.Parameter(torch.randn(hidden_size))
        
        # Value estimation layers
        self.dense1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, 1)
        
        # Pattern recognition components
        self.pattern_dense = nn.Linear(hidden_size, hidden_size // 4)
        self.pattern_classifier = nn.Linear(hidden_size // 4, 1)
        
        # Convergence tracking
        self.prev_values = None
        self.convergence_threshold = getattr(config, 'convergence_threshold', 0.005) if config else 0.005
        self.convergence_patience = 2  # Number of iterations with minimal change to confirm convergence
        self.convergence_counter = 0
        
        # Value history tracking for visualization
        self.value_history = []
    
    def attention_pooling(self, hidden_states):
        """Apply attention-weighted pooling to the hidden states."""
        # Calculate attention scores
        query = self.attention_query.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        scores = torch.matmul(hidden_states, query.transpose(-1, -2))  # [batch, seq_len, 1]
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1)
        
        # Apply weights to hidden states
        weighted = torch.matmul(weights.transpose(-1, -2), hidden_states)  # [batch, 1, hidden_size]
        
        return weighted.squeeze(1)  # [batch, hidden_size]
    
    def detect_pattern(self, hidden_states):
        """Detect structured patterns in hidden states."""
        # Apply pattern detection to each token position
        batch_size, seq_len, _ = hidden_states.shape
        pattern_scores = []
        
        for i in range(seq_len):
            x = self.pattern_dense(hidden_states[:, i])
            x = F.relu(x)
            pattern_score = torch.sigmoid(self.pattern_classifier(x))
            pattern_scores.append(pattern_score)
        
        # Combine pattern scores
        combined_score = torch.stack(pattern_scores, dim=1).mean(dim=1)
        return combined_score
    
    def forward(self, hidden_states):
        """Estimate the value of the hidden states."""
        # Apply attention pooling
        pooled = self.attention_pooling(hidden_states)
        
        # Get pattern score
        pattern_score = self.detect_pattern(hidden_states)
        
        # Project to scalar value
        x = self.dense1(pooled)
        x = F.relu(x)
        x = self.dense2(x)
        base_value = torch.sigmoid(x)  # Normalize to [0, 1]
        
        # Combine base value with pattern recognition
        value = 0.7 * base_value + 0.3 * pattern_score
        
        # Record value in history
        self.value_history.append(value.detach().mean().item())
        
        return value
    
    def check_convergence(self, hidden_states):
        """
        Check if the value has converged (minimal change over iterations).
        
        Args:
            hidden_states: Hidden state tensor
            
        Returns:
            True if converged, False otherwise
        """
        # Get current value estimate
        current_value = self.forward(hidden_states).mean().item()
        
        # Initialize previous value if needed
        if self.prev_values is None:
            self.prev_values = current_value
            return False
        
        # Calculate change in value
        value_change = abs(current_value - self.prev_values)
        
        # Update previous value
        self.prev_values = current_value
        
        # Check for convergence
        if value_change < self.convergence_threshold:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_patience:
                return True
        else:
            # Reset counter if change is significant
            self.convergence_counter = 0
        
        return False
    
    def should_continue_iteration(self, hidden_states, current_iteration, min_iterations, max_iterations):
        """
        Determine whether to continue iterating based on value estimates and iteration limits.
        
        Args:
            hidden_states: Current hidden states
            current_iteration: Current iteration count
            min_iterations: Minimum iterations to perform
            max_iterations: Maximum iterations to perform
            
        Returns:
            True if iteration should continue, False otherwise
        """
        # Always do at least min_iterations
        if current_iteration < min_iterations:
            return True
        
        # Never exceed max_iterations
        if current_iteration >= max_iterations:
            return False
        
        # Check for convergence
        return not self.check_convergence(hidden_states)
    
    def reset(self):
        """Reset convergence tracking and history"""
        self.prev_values = None
        self.convergence_counter = 0
        self.value_history = []
    
    def get_value_history(self):
        """Get the history of value estimates"""
        return self.value_history