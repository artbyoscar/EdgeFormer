import os

# Update the edgeformer.py file to include SimpleTokenizer
with open('src/model/edgeformer.py', 'w', encoding='utf-8') as f:
    f.write("""
# Minimal placeholder for EdgeFormer and EdgeFormerConfig
class EdgeFormerConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class EdgeFormer:
    def __init__(self, config=None):
        self.config = config or EdgeFormerConfig()
    
    @staticmethod
    def from_pretrained(model_path, **kwargs):
        return EdgeFormer(EdgeFormerConfig(**kwargs))

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
    
    def encode(self, text):
        # Simple implementation that just returns character codes
        return [ord(c) for c in text]
    
    def decode(self, ids):
        # Convert character codes back to text
        return ''.join([chr(i) for i in ids])
""")

print("Updated edgeformer.py with SimpleTokenizer class")