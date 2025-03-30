import unittest
import torch
from memory.htps_memory import HTPSMemory
from memory.retriever import MemoryRetriever

class TestHTPSMemory(unittest.TestCase):
    def setUp(self):
        self.memory = HTPSMemory(capacity=5, hidden_size=64, selection_strategy='htps')
        self.retriever = MemoryRetriever(hidden_size=64)
        
    def test_add_memory(self):
        """Test adding a memory works correctly."""
        text = "Test memory text"
        result = self.memory.add_memory(text)
        self.assertTrue(result)
        self.assertEqual(len(self.memory.list_memories()), 1)
        self.assertEqual(self.memory.list_memories()[0], text)
        
    def test_memory_capacity(self):
        """Test that capacity limit is respected."""
        for i in range(10):  # Add more than capacity
            self.memory.add_memory(f"Memory {i}")
        
        # Should only have the most recent 5
        self.assertEqual(len(self.memory.list_memories()), 5)
        self.assertEqual(self.memory.list_memories()[0], "Memory 5")
        
    def test_memory_retrieval(self):
        """Test that memory retrieval works."""
        for i in range(5):
            self.memory.add_memory(f"Memory {i}")
            
        # Create a test query
        query = torch.randn(1, 1, 64)
        
        # Get memories
        memory_vectors, attention_weights, memory_texts = self.retriever.retrieve_memories(
            query, self.memory, top_k=3
        )
        
        # Should have 3 memories retrieved
        self.assertEqual(len(memory_texts), 3)

if __name__ == '__main__':
    unittest.main()