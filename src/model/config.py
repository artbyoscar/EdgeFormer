class EdgeFormerConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        latent_size_factor=8,  # Latent size divisor for MLA
        num_kv_groups=4,  # For grouped-query attention
        use_sliding_window=True,
        sliding_window_size=512,
        use_flash_attention=True,
        use_sparse_mlp=True,
        mlp_sparsity=0.8,
        quantization="int8",  # Options: None, "int8", "int4"
        optimize_for_rdna3=True,  # RDNA3-specific optimizations
        debug_mode=False,  # For logging
        # Budget forcing settings
        enable_budget_forcing=False,
        max_budget_tokens=2048,
        max_thinking_extensions=2,
        extension_token="Wait",
        budget_criteria="balanced",
        # Value-based recurrent depth settings
        enable_recurrent_depth=False,
        max_iterations=32,
        convergence_threshold=0.01,
        adaptive_iterations=True,
    ):
        """
        Initialize EdgeFormerConfig with standard parameters and HyperTree-inspired budget forcing settings.
        
        Args:
            # Standard parameters
            vocab_size: Size of the vocabulary
            hidden_size: Size of the hidden states
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            intermediate_size: Size of the intermediate layer
            hidden_act: Activation function for hidden layers
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention probabilities
            max_position_embeddings: Maximum supported sequence length
            type_vocab_size: Size of the token type vocabulary
            initializer_range: Range for weight initialization
            layer_norm_eps: Epsilon for layer normalization
            pad_token_id: ID for padding token
            bos_token_id: ID for beginning of sequence token
            eos_token_id: ID for end of sequence token
            
            # MLA and optimization specific
            latent_size_factor: Divisor for latent size calculation
            num_kv_groups: Number of KV groups for grouped-query attention
            use_sliding_window: Whether to use sliding window attention
            sliding_window_size: Size of attention window
            use_flash_attention: Whether to use FlashAttention
            use_sparse_mlp: Whether to use sparse MLP
            mlp_sparsity: Sparsity factor for MLP
            quantization: Quantization type (None, "int8", "int4")
            optimize_for_rdna3: Whether to use RDNA3-specific optimizations
            debug_mode: Enable detailed logging
            
            # Budget forcing settings
            enable_budget_forcing: Whether to enable HyperTree budget forcing
            max_budget_tokens: Maximum tokens before budget enforcement
            max_thinking_extensions: Maximum number of thinking extensions
            extension_token: Token to insert for thinking extension
            budget_criteria: Path selection criteria ("speed", "quality", or "balanced")
            
            # Value-based recurrent depth settings
            enable_recurrent_depth: Whether to enable value-based recurrent depth processing
            max_iterations: Maximum number of iterations for recurrent processing
            convergence_threshold: Threshold for determining convergence
            adaptive_iterations: Whether to adapt iteration count based on task complexity
        """
        # Standard parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # MLA and optimization specific
        self.latent_size_factor = latent_size_factor
        self.latent_size = hidden_size // latent_size_factor
        self.num_kv_groups = num_kv_groups
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.use_flash_attention = use_flash_attention
        self.use_sparse_mlp = use_sparse_mlp
        self.mlp_sparsity = mlp_sparsity
        self.quantization = quantization
        self.optimize_for_rdna3 = optimize_for_rdna3
        self.debug_mode = debug_mode
        
        # Budget forcing settings
        self.enable_budget_forcing = enable_budget_forcing
        self.max_budget_tokens = max_budget_tokens
        self.max_thinking_extensions = max_thinking_extensions
        self.extension_token = extension_token
        self.budget_criteria = budget_criteria
        
        # Value-based recurrent depth settings
        self.enable_recurrent_depth = enable_recurrent_depth
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.adaptive_iterations = adaptive_iterations
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate MLA-specific settings
        if not self.latent_size > 0:
            raise ValueError("latent_size must be greater than 0")
        
        # Validate sliding window settings
        if self.use_sliding_window and (self.sliding_window_size <= 0 or self.sliding_window_size > self.max_position_embeddings):
            raise ValueError(f"sliding_window_size must be between 1 and {self.max_position_embeddings}")
        
        # Validate sparsity settings
        if self.use_sparse_mlp and (self.mlp_sparsity <= 0 or self.mlp_sparsity >= 1):
            raise ValueError("mlp_sparsity must be between 0 and 1 (exclusive)")
        
        # Validate budget forcing settings
        if self.enable_budget_forcing:
            if self.max_budget_tokens > self.max_position_embeddings:
                raise ValueError("max_budget_tokens cannot exceed max_position_embeddings")
                
            if self.max_thinking_extensions <= 0:
                raise ValueError("max_thinking_extensions must be greater than 0")
                
            if self.budget_criteria not in ["speed", "quality", "balanced"]:
                raise ValueError("budget_criteria must be one of: 'speed', 'quality', or 'balanced'")
        
        # Validate recurrent depth settings
        if self.enable_recurrent_depth:
            if self.max_iterations <= 0:
                raise ValueError("max_iterations must be greater than 0")
                
            if self.convergence_threshold <= 0:
                raise ValueError("convergence_threshold must be greater than 0")