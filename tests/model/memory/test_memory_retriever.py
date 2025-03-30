# tests/model/test_memory_retriever.py
import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.memory_integration.memory_retriever import MemoryRetriever

class MockMemory:
    """Mock memory class for testing."""
    
    def __init__(self, num_entries=10, vector_size=128):
        self.vectors = torch.randn(num_entries, vector_size)
        self.texts = [f"Memory entry {i}" for i in range(num_entries)]
        self.is_empty_val = False
    
    def is_empty(self):
        return self.is_empty_val
    
    def get_vectors(self):
        return self.vectors
    
    def get_texts(self):
        return self.texts
    
    def set_empty(self, empty):
        self.is_empty_val = empty

class TestMemoryRetriever(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 768
        self.projection_size = 128
        self.retriever = MemoryRetriever(
            hidden_size=self.hidden_size,
            memory_projection_size=self.projection_size,
            temperature=0.5,
            retrieval_threshold=0.5,
        )
        self.memory = MockMemory(num_entries=10, vector_size=self.projection_size)
        self.batch_size = 2
        self.query_hidden = torch.rand(self.batch_size, 1, self.hidden_size)
    
    def test_projection(self):
        """Test if projection works correctly."""
        projected = self.retriever.project_query(self.query_hidden)
        self.assertEqual(projected.shape, (self.batch_size, 1, self.projection_size))
    
    def test_empty_memory(self):
        """Test retrieval with empty memory."""
        self.memory.set_empty(True)
        memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
            self.query_hidden, self.memory, top_k=3
        )
        self.assertIsNone(memory_vectors)
        self.assertIsNone(attention_weights)
        self.assertIsNone(memory_texts)
    
    def test_retrieval(self):
        """Test normal memory retrieval."""
        top_k = 3
        memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
            self.query_hidden, self.memory, top_k=top_k
        )
    
        # Check if memory vectors is the original memory vectors
        self.assertTrue(torch.equal(memory_vectors, self.memory.get_vectors()))
    
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, 10))
    
        # Check weights sum to 1
        self.assertTrue(torch.allclose(torch.sum(attention_weights, dim=1), torch.ones(self.batch_size)))
    
        # Count positive values per row
        for b in range(self.batch_size):
            positive_count = torch.sum(attention_weights[b] > 0).item()
            self.assertLessEqual(positive_count, top_k,
                                 f"Row {b} has {positive_count} positive values, expected <= {top_k}")
    
    def test_attention_capture(self):
        """Test capturing attention for visualization."""
        memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
            self.query_hidden, self.memory, top_k=3, capture_attention=True
        )
        
        # Should have text representations
        self.assertIsNotNone(memory_texts)
        self.assertEqual(len(memory_texts), self.batch_size)
        
        # Each entry should be (text, weight) tuple
        self.assertTrue(all(isinstance(entry, tuple) for batch in memory_texts for entry in batch))
        
        # Weights in tuples should match attention weights
        for b in range(self.batch_size):
            for text, weight in memory_texts[b]:
                # Find corresponding index
                idx = self.memory.get_texts().index(text)
                self.assertAlmostEqual(weight, attention_weights[b, idx].item(), places=5)

if __name__ == "__main__":
    unittest.main()