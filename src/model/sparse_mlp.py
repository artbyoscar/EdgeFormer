import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.sparsity = config.mlp_sparsity
        
        # FFN layers
        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize sparsity masks
        self._initialize_sparsity_masks()
    
    def _initialize_sparsity_masks(self):
        """Initialize random sparsity masks for the FFN layers."""
        # Create weight sparsity masks during initialization
        # These are 2D masks for the weight matrices
        self.register_buffer(
            "up_mask", 
            torch.bernoulli(torch.ones(self.intermediate_size, self.hidden_size) * (1 - self.sparsity))
        )
        self.register_buffer(
            "down_mask", 
            torch.bernoulli(torch.ones(self.hidden_size, self.intermediate_size) * (1 - self.sparsity))
        )
        
        # Runtime activation masks will be created during forward pass
    
    def apply_sparsity_masks(self):
        """Apply sparsity masks to weights."""
        # For up projection (hidden_size -> intermediate_size)
        if hasattr(self, 'up_mask') and self.up_mask.shape == self.dense_h_to_4h.weight.shape:
            self.dense_h_to_4h.weight.data *= self.up_mask
        
        # For down projection (intermediate_size -> hidden_size)
        if hasattr(self, 'down_mask') and self.down_mask.shape == self.dense_4h_to_h.weight.shape:
            self.dense_4h_to_h.weight.data *= self.down_mask

    def get_activation_mask(self, shape, device):
        """
        Generate activation sparsity mask with the correct dimensions.
        
        Args:
            shape: Tuple (batch_size, seq_length, dim)
            device: The device to create the mask on
            
        Returns:
            A sparsity mask tensor of shape (batch_size, seq_length, dim)
        """
        batch_size, seq_length, dim = shape
        # Create a fresh mask with proper dimensions for the activations
        mask = torch.bernoulli(
            torch.ones(batch_size, seq_length, dim, device=device) * (1 - self.sparsity)
        )
        return mask
    
    def forward(self, hidden_states):
        # Apply weight sparsity masks
        if self.training:
            self.apply_sparsity_masks()
        
        # Forward pass
        intermediate = self.dense_h_to_4h(hidden_states)
        
        # Apply activation sparsity if needed
        if self.sparsity > 0:
            # Create activation mask with correct dimensions
            act_mask = self.get_activation_mask(
                intermediate.shape, 
                intermediate.device
            )
            # Apply mask before activation
            intermediate = intermediate * act_mask
        
        # Apply activation function
        intermediate = F.gelu(intermediate)
        
        # Down projection
        output = self.dense_4h_to_h(intermediate)
        output = self.dropout(output)
        
        return output