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
        debug_mode=False,  # Add this parameter for logging
    ):
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
        self.debug_mode = debug_mode  # Add this line to store the debug mode parameter