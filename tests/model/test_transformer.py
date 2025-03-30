#!/usr/bin/env python
# tests/model/test_transformer.py
import unittest
import torch
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.transformer.config import EdgeFormerConfig
from src.model.transformer.base_transformer import (
    EdgeFormerEmbeddings,
    EdgeFormerSelfAttention,
    EdgeFormerAttention,
    EdgeFormerLayer,
    EdgeFormerEncoder,
    EdgeFormerModel,
    EdgeFormer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test_transformer")

class TestEdgeFormerComponents(unittest.TestCase):
    """Test EdgeFormer components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EdgeFormerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            attention_type="standard",
            sliding_window_size=128,
        )
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Test inputs
        self.batch_size = 2
        self.seq_length = 16
        self.input_ids = torch.randint(
            0, self.config.vocab_size, (self.batch_size, self.seq_length), device=self.device
        )
        self.attention_mask = torch.ones(
            (self.batch_size, self.seq_length), device=self.device
        )
    
    def test_embeddings(self):
        """Test EdgeFormerEmbeddings."""
        logger.info("Testing embeddings...")
        embeddings = EdgeFormerEmbeddings(self.config).to(self.device)
        
        # Forward pass
        outputs = embeddings(self.input_ids)
        
        # Check output shape
        self.assertEqual(
            outputs.shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        logger.info("Embeddings test passed")
    
    def test_self_attention(self):
        """Test EdgeFormerSelfAttention."""
        logger.info("Testing self-attention...")
        self_attention = EdgeFormerSelfAttention(self.config).to(self.device)
        
        # Create hidden states
        hidden_states = torch.rand(
            (self.batch_size, self.seq_length, self.config.hidden_size), 
            device=self.device
        )
        
        # Create attention mask
        attention_mask = (1.0 - self.attention_mask[:, None, None, :]) * -10000.0
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self_attention(hidden_states, attention_mask)
        
        # Check output shape
        self.assertEqual(
            outputs[0].shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Check past key value
        self.assertIsNotNone(outputs[-1])
        
        logger.info("Self-attention test passed")
    
    def test_full_attention(self):
        """Test EdgeFormerAttention."""
        logger.info("Testing full attention...")
        attention = EdgeFormerAttention(self.config).to(self.device)
        
        # Create hidden states
        hidden_states = torch.rand(
            (self.batch_size, self.seq_length, self.config.hidden_size), 
            device=self.device
        )
        
        # Create attention mask
        attention_mask = (1.0 - self.attention_mask[:, None, None, :]) * -10000.0
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = attention(hidden_states, attention_mask)
        
        # Check output shape
        self.assertEqual(
            outputs[0].shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        logger.info("Full attention test passed")
    
    def test_transformer_layer(self):
        """Test EdgeFormerLayer."""
        logger.info("Testing transformer layer...")
        layer = EdgeFormerLayer(self.config).to(self.device)
        
        # Create hidden states
        hidden_states = torch.rand(
            (self.batch_size, self.seq_length, self.config.hidden_size), 
            device=self.device
        )
        
        # Create attention mask
        attention_mask = (1.0 - self.attention_mask[:, None, None, :]) * -10000.0
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = layer(hidden_states, attention_mask)
        
        # Check output shape
        self.assertEqual(
            outputs[0].shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        logger.info("Transformer layer test passed")
    
    def test_transformer_encoder(self):
        """Test EdgeFormerEncoder."""
        logger.info("Testing transformer encoder...")
        encoder = EdgeFormerEncoder(self.config).to(self.device)
        
        # Create hidden states
        hidden_states = torch.rand(
            (self.batch_size, self.seq_length, self.config.hidden_size), 
            device=self.device
        )
        
        # Create attention mask
        attention_mask = (1.0 - self.attention_mask[:, None, None, :]) * -10000.0
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = encoder(
            hidden_states, 
            attention_mask, 
            output_hidden_states=True, 
            output_attentions=True
        )
        
        # Check output shapes
        self.assertEqual(
            outputs[0].shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Check hidden states
        self.assertEqual(len(outputs[1]), self.config.num_hidden_layers + 1)
        
        logger.info("Transformer encoder test passed")
    
    def test_transformer_model(self):
        """Test EdgeFormerModel."""
        logger.info("Testing transformer model...")
        model = EdgeFormerModel(self.config).to(self.device)
        
        # Forward pass
        outputs = model(
            self.input_ids,
            attention_mask=self.attention_mask,
            output_hidden_states=True,
        )
        
        # Check output shape
        self.assertEqual(
            outputs[0].shape, 
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Check hidden states
        self.assertIsNotNone(outputs[1])
        
        logger.info("Transformer model test passed")
    
    def test_full_edgeformer(self):
        """Test complete EdgeFormer."""
        logger.info("Testing complete EdgeFormer...")
        model = EdgeFormer(self.config).to(self.device)
        
        # Forward pass
        outputs = model(
            self.input_ids,
            attention_mask=self.attention_mask,
        )
        
        # Check output shape
        self.assertEqual(
            outputs[0].shape, 
            (self.batch_size, self.seq_length, self.config.vocab_size)
        )
        
        logger.info("Complete EdgeFormer test passed")
    
    def test_generation(self):
        """Test text generation."""
        logger.info("Testing text generation...")
        model = EdgeFormer(self.config).to(self.device)
        
        # Input for generation
        input_text = torch.randint(
            0, self.config.vocab_size, (1, 4), device=self.device
        )
        
        # Generate text
        generated = model.generate(
            input_text,
            max_length=10,
            do_sample=False,  # Greedy decoding
        )
        
        # Check output shape
        self.assertGreaterEqual(generated.shape[1], input_text.shape[1])
        self.assertLessEqual(generated.shape[1], 10)
        
        logger.info("Text generation test passed")
    
    def test_attention_types(self):
        """Test different attention types."""
        attention_types = ["standard", "sliding_window"]
        for attention_type in attention_types:
            logger.info(f"Testing {attention_type} attention...")
            
            # Create config with specific attention type
            config = EdgeFormerConfig(
                vocab_size=1000,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=512,
                max_position_embeddings=512,
                attention_type=attention_type,
                sliding_window_size=8,
            )
            
            # Create model with this attention type
            model = EdgeFormer(config).to(self.device)
            
            # Forward pass
            outputs = model(
                self.input_ids,
                attention_mask=self.attention_mask,
            )
            
            # Check output shape
            self.assertEqual(
                outputs[0].shape, 
                (self.batch_size, self.seq_length, config.vocab_size)
            )
            
            logger.info(f"{attention_type} attention test passed")

if __name__ == "__main__":
    unittest.main()