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
        self.num_key_value_heads = num_key_value_heads
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
            
        # Set default KV heads for GQA if not provided (safety check)
        if self.num_key_value_heads is None and attention_type == "gqa":
            self.num_key_value_heads = max(1, self.num_attention_heads // 4)
        
        # Validate configuration at the end of initialization
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        # Validate basic attention head configuration
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Validate GQA configuration
        if self.attention_type == "gqa":
            if self.num_key_value_heads is None:
                raise ValueError("num_key_value_heads must be specified when using GQA attention")
            
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ValueError(
                    f"For GQA: num_attention_heads ({self.num_attention_heads}) must be "
                    f"divisible by num_key_value_heads ({self.num_key_value_heads})"
                )
            
            if self.num_key_value_heads > self.num_attention_heads:
                raise ValueError(
                    f"num_key_value_heads ({self.num_key_value_heads}) cannot be greater than "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            
            if self.num_key_value_heads <= 0:
                raise ValueError(f"num_key_value_heads must be positive, got {self.num_key_value_heads}")
        
        # Validate MLA configuration
        if self.attention_type == "mla":
            if self.latent_size is not None and self.latent_size <= 0:
                raise ValueError(f"latent_size must be positive, got {self.latent_size}")
            
            if self.latent_size is not None and self.latent_size > self.hidden_size:
                raise ValueError(
                    f"latent_size ({self.latent_size}) cannot be greater than "
                    f"hidden_size ({self.hidden_size})"
                )
        
        # Validate other parameters
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {self.intermediate_size}")
        
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {self.max_position_embeddings}")
        
        if not 0 <= self.hidden_dropout_prob <= 1:
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1, got {self.hidden_dropout_prob}")
        
        if not 0 <= self.attention_probs_dropout_prob <= 1:
            raise ValueError(f"attention_probs_dropout_prob must be between 0 and 1, got {self.attention_probs_dropout_prob}")
        
        # Validate attention type
        valid_attention_types = ["standard", "mla", "gqa", "sliding_window"]
        if self.attention_type not in valid_attention_types:
            raise ValueError(
                f"attention_type must be one of {valid_attention_types}, got '{self.attention_type}'"
            )
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create an EdgeFormerConfig from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__.copy()
    
    def __repr__(self):
        """String representation of the configuration."""
        return f"EdgeFormerConfig({self.to_dict()})"
    
    def get_attention_info(self):
        """Get attention configuration information."""
        info = {
            "attention_type": self.attention_type,
            "num_attention_heads": self.num_attention_heads,
            "hidden_size": self.hidden_size,
            "head_dim": self.hidden_size // self.num_attention_heads,
        }
        
        if self.attention_type == "gqa":
            info.update({
                "num_key_value_heads": self.num_key_value_heads,
                "queries_per_kv_head": self.num_attention_heads // self.num_key_value_heads,
            })
        
        if self.attention_type == "mla":
            info.update({
                "latent_size": self.latent_size,
            })
        
        if self.attention_type == "sliding_window":
            info.update({
                "sliding_window_size": self.sliding_window_size,
            })
        
        return info
