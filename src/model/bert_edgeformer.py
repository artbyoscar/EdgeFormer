# Create: src/model/bert_edgeformer.py
class BERTEdgeFormer(EdgeFormer):
    """EdgeFormer optimized for BERT-family models"""
    
    def __init__(self, config):
        super().__init__(config)
        # BERT-specific optimizations
        self.bert_sensitive_layers = [
            'embeddings.word_embeddings',
            'embeddings.position_embeddings', 
            'embeddings.token_type_embeddings',
            'pooler.dense'
        ]
    
    def get_compression_strategy(self):
        """BERT-optimized compression approach"""
        return {
            "embedding_preservation": True,
            "attention_head_grouping": True,
            "layer_norm_preservation": True
        }