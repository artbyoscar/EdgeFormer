# Improved version of ValueEstimator - could be put in src/utils/improved_value_estimator.py
import torch
import torch.nn as nn

class ImprovedValueEstimator(nn.Module):
    """
    Enhanced value estimator with better pattern recognition for structured vs random states.
    """
    def __init__(self, hidden_size, config=None):
        super().__init__()
        
        # Richer intermediate representations
        intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4) if config else hidden_size * 4
        
        # Use a more sophisticated pooling mechanism
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Deeper value network with residual connections
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size // 2),
            nn.GELU(),
            nn.Linear(intermediate_size // 2, intermediate_size // 2),
            nn.GELU(),
            nn.Linear(intermediate_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Pattern detection features
        self.pattern_features = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size // 4),
            nn.GELU(),
            nn.Linear(intermediate_size // 4, intermediate_size // 4)
        )
        
        # Convergence tracking with adaptive threshold
        self.prev_values = None
        self.convergence_threshold = 0.005
        self.min_convergence_threshold = 0.001
        self.max_convergence_threshold = 0.01
        self.convergence_patience = 3
        self.convergence_counter = 0
        self.value_history = []
        self.change_history = []
        
    def forward(self, hidden_states):
        """
        Compute value estimate with attention-weighted pooling and pattern detection.
        
        Args:
            hidden_states: Hidden state tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor of shape [batch_size, 1] with value estimates
        """
        # Attention-weighted pooling
        attn_weights = self.attn_pool(hidden_states)
        pooled = torch.sum(hidden_states * attn_weights, dim=1)
        
        # Extract pattern features
        pattern_features = self.pattern_features(pooled)
        
        # Pattern coherence - higher for structured patterns
        batch_size = pattern_features.size(0)
        coherence = torch.zeros(batch_size, 1, device=pattern_features.device)
        
        if batch_size > 1:
            # Measure similarity between examples in batch (structured data should be more similar)
            pattern_norm = nn.functional.normalize(pattern_features, p=2, dim=1)
            similarity = torch.mm(pattern_norm, pattern_norm.t())
            # Average similarity excluding self-similarity
            mask = torch.ones_like(similarity) - torch.eye(batch_size, device=similarity.device)
            coherence = (similarity * mask).sum(dim=1, keepdim=True) / (batch_size - 1)
        
        # Compute value with influence from coherence
        value = self.value_net(pooled) * (1.0 + 0.2 * coherence)
        value = torch.clamp(value, 0.0, 1.0)
        
        # Record value in history
        value_mean = value.detach().mean().item()
        self.value_history.append(value_mean)
        
        return value
    
    def estimate_confidence(self, hidden_states):
        """
        Estimate confidence score with enhanced pattern detection.
        
        Args:
            hidden_states: Hidden state tensor
            
        Returns:
            Confidence score (0-1)
        """
        return self.forward(hidden_states).mean().item()
    
    def check_convergence(self, hidden_states):
        """
        Check if the value has converged with adaptive threshold.
        
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
        self.change_history.append(value_change)
        
        # Adaptive threshold based on recent history
        if len(self.change_history) > 5:
            recent_changes = self.change_history[-5:]
            avg_change = sum(recent_changes) / len(recent_changes)
            # Adjust threshold based on average change
            self.convergence_threshold = max(
                self.min_convergence_threshold,
                min(self.max_convergence_threshold, avg_change * 0.5)
            )
        
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
        Enhanced decision for iteration continuation with additional checks.
        
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
        converged = self.check_convergence(hidden_states)
        
        # Check for value plateauing or declining
        if len(self.value_history) >= 3:
            recent_values = self.value_history[-3:]
            # If values are declining after reaching a good level, stop
            if current_iteration > min_iterations + 5 and recent_values[0] > recent_values[-1]:
                # Only stop if the value is reasonably high
                if recent_values[0] > 0.6:
                    return False
        
        return not converged
    
    def reset(self):
        """Reset convergence tracking and history"""
        self.prev_values = None
        self.convergence_counter = 0
        self.value_history = []
        self.change_history = []
    
    def get_value_history(self):
        """Get the history of value estimates"""
        return self.value_history
    
    def get_change_history(self):
        """Get the history of value changes between iterations"""
        return self.change_history