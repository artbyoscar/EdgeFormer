# Create: src/model/vit_edgeformer.py
class ViTEdgeFormer(EdgeFormer):
    """EdgeFormer for Vision Transformers"""
    
    def __init__(self, config):
        super().__init__(config)
        self.vit_sensitive_layers = [
            'patch_embed',
            'pos_embed',
            'cls_token',
            'head'
        ]
    
    def patch_aware_quantization(self, layer):
        """Special handling for patch embedding layers"""
        # Vision-specific quantization strategies
