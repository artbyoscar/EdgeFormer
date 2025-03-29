import os
import re

def integrate_kv_cache_manager():
    """
    Integrate KV Cache Manager with EdgeFormer
    """
    print("Integrating KV Cache Manager with EdgeFormer...")
    
    edgeformer_path = "src/model/edgeformer.py"
    
    # Check if the file exists
    if not os.path.exists(edgeformer_path):
        print(f"Error: {edgeformer_path} not found")
        return False
    
    # Read the file
    with open(edgeformer_path, "r") as f:
        content = f.read()
    
    # Import KV Cache Manager
    if "from src.utils.kv_cache_manager import KVCacheManager" not in content:
        import_section = re.search(r"import.*?\n\n", content, re.DOTALL)
        if import_section:
            updated_imports = import_section.group(0).strip() + "\nfrom src.utils.kv_cache_manager import KVCacheManager\n\n"
            content = content.replace(import_section.group(0), updated_imports)
    
    # Add KV Cache initialization to __init__
    init_pattern = r"def __init__\(self, config\):.*?self\.config = config"
    if "__init__" in content and "self.kv_cache_manager = None" not in content:
        init_section = re.search(init_pattern, content, re.DOTALL)
        if init_section:
            updated_init = init_section.group(0) + "\n        self.kv_cache_manager = None"
            content = content.replace(init_section.group(0), updated_init)
    
    # Initialize KV Cache Manager in forward method
    forward_pattern = r"def forward\(self, input_ids.*?\):"
    forward_section = re.search(forward_pattern, content)
    
    if forward_section and "if self.kv_cache_manager is None:" not in content:
        # Add code to initialize KV Cache Manager
        forward_append = """
        # Initialize KV Cache Manager if needed
        if self.kv_cache_manager is None:
            self.kv_cache_manager = KVCacheManager(
                max_batch_size=1,  # Adjust based on batch size
                max_seq_length=self.config.max_position_embeddings,
                num_layers=self.config.num_hidden_layers,
                num_heads=self.config.num_attention_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                device=input_ids.device
            )
        """
        
        # Insert after forward method declaration
        content = content.replace(
            forward_section.group(0),
            forward_section.group(0) + forward_append
        )
    
    # Modify attention calculation to use KV Cache
    attention_pattern = r"def self_attention\(.*?\):"
    attention_section = re.search(attention_pattern, content, re.DOTALL)
    
    if attention_section and "# Use KV Cache for attention" not in content:
        # Find where key and value are calculated in the function
        key_value_pattern = r"key = self\.key\(hidden_states\).*?value = self\.value\(hidden_states\)"
        key_value_section = re.search(key_value_pattern, content, re.DOTALL)
        
        if key_value_section:
            # Create updated code to use KV Cache
            updated_key_value = """
            # Regular K/V computation
            key = self.key(hidden_states)
            value = self.value(hidden_states)
            
            # Use KV Cache for attention
            layer_idx = self.layer_idx if hasattr(self, 'layer_idx') else 0
            
            # Update KV Cache with new key and value
            if self.kv_cache_manager is not None:
                self.kv_cache_manager.update(layer_idx, key, value)
                
                # Get full cached sequence for attention
                cached_key, cached_value = self.kv_cache_manager.get(layer_idx)
                key = cached_key
                value = cached_value
            """
            
            content = content.replace(key_value_section.group(0), updated_key_value)
    
    # Add layer_idx to attention layers initialization
    init_attention_pattern = r"self\.attention = nn\.ModuleList\(\[.*?\]\)"
    init_attention_section = re.search(init_attention_pattern, content, re.DOTALL)
    
    if init_attention_section and "layer_idx=" not in content:
        updated_attention = init_attention_section.group(0).replace(
            "Self_Attention(config)",
            "Self_Attention(config, layer_idx=i)"
        )
        content = content.replace(init_attention_section.group(0), updated_attention)
    
    # Add layer_idx to Self_Attention __init__
    self_attention_init = r"class Self_Attention\(nn\.Module\):.*?def __init__\(self, config\):"
    self_attention_init_section = re.search(self_attention_init, content, re.DOTALL)
    
    if self_attention_init_section and "layer_idx=0" not in content:
        updated_sa_init = self_attention_init_section.group(0).replace(
            "def __init__(self, config):",
            "def __init__(self, config, layer_idx=0):"
        )
        content = content.replace(self_attention_init_section.group(0), updated_sa_init)
        
        # Add layer_idx to instance variables
        sa_init_body = r"def __init__\(self, config.*?\):.*?self\.dropout = nn\.Dropout\(config\.hidden_dropout_prob\)"
        sa_init_body_section = re.search(sa_init_body, content, re.DOTALL)
        
        if sa_init_body_section and "self.layer_idx = layer_idx" not in content:
            updated_sa_init_body = sa_init_body_section.group(0) + "\n        self.layer_idx = layer_idx"
            content = content.replace(sa_init_body_section.group(0), updated_sa_init_body)
    
    # Write updated content back to file
    with open(edgeformer_path, "w") as f:
        f.write(content)
    
    print("KV Cache Manager integration complete!")
    return True

if __name__ == "__main__":
    integrate_kv_cache_manager()