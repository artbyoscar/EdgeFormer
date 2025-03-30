"""Configuration for EdgeFormer Transformer model."""

class EdgeFormerConfig:
    """Configuration class for EdgeFormer model."""
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,  # For GQA
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        eos_token_id=50256,
        attention_type="standard",  # "standard", "mla", "gqa", "sliding_window"
        sliding_window_size=512,
        use_cache=True,
        use_memory=False,
        memory_capacity=100,
        memory_strategy="htps",
        # MLA specific
        latent_size=None,  # If None, will default to hidden_size // 4
    ):
        """Initialize configuration with default values."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads  # Add this
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.attention_type = attention_type
        self.sliding_window_size = sliding_window_size
        self.use_cache = use_cache
        self.use_memory = use_memory
        self.memory_capacity = memory_capacity
        self.memory_strategy = memory_strategy
        
        # Set MLA latent size if not provided
        if latent_size is None and attention_type == "mla":
            self.latent_size = hidden_size // 4
        else:
            self.latent_size = latent_size
            
        # Set GQA key-value heads if not provided
        if num_key_value_heads is None and attention_type == "gqa":
            self.num_key_value_heads = max(1, num_attention_heads // 4)
        else:
            self.num_key_value_heads = num_key_value_heads
            
        # Set default KV heads for GQA if not provided
        if self.num_key_value_heads is None and attention_type == "gqa":
            self.num_key_value_heads = max(1, self.num_attention_heads // 4)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a EdgeFormerConfig from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()