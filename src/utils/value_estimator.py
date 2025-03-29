import torch
import torch.nn as nn

class ValueEstimator(nn.Module):
    """
    Estimates the value of the current state for intelligent stopping based on diminishing returns.
    """
    def __init__(self, hidden_size, config=None):
        """
        Initialize the value estimator.
        
        Args:
            hidden_size: Size of the hidden representations
            config: Optional configuration object
        """
        super().__init__()
        
        # Default intermediate size if not specified
        intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4) if config else hidden_size * 4
        
        # Value estimation network
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size // 4),
            nn.GELU(),
            nn.Linear(intermediate_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Convergence tracking
        self.prev_values = None
        self.convergence_threshold = 0.005  # Default threshold
        self.convergence_patience = 3  # Number of iterations with minimal change to confirm convergence
        self.convergence_counter = 0
        
        # Value history tracking for visualization
        self.value_history = []
    
    def forward(self, hidden_states):
        """
        Compute value estimate for the given hidden states.
        
        Args:
            hidden_states: Hidden state tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor of shape [batch_size, 1] with value estimates
        """
        # Average pooling across sequence dimension
        pooled = hidden_states.mean(dim=1)
        
        # Compute value
        value = self.value_head(pooled)
        
        # Record value in history
        self.value_history.append(value.detach().mean().item())
        
        return value
    
    def estimate_confidence(self, hidden_states):
        """
        Estimate confidence score (higher value = more confident).
        For use with budget forcing.
        
        Args:
            hidden_states: Hidden state tensor
            
        Returns:
            Confidence score (0-1)
        """
        return self.forward(hidden_states).mean().item()
    
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