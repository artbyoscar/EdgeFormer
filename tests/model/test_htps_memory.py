import unittest
import torch
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.model.associative_memory.htps_memory import HTPSMemory
from src.model.associative_memory.memory_retriever import MemoryRetriever

class TestHTPSMemory(unittest.TestCase):
    def setUp(self):
        self.memory = HTPSMemory(capacity=5, hidden_size=64, selection_strategy='htps')
        
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
        
    def test_clear_memories(self):
        """Test clearing memories."""
        for i in range(3):
            self.memory.add_memory(f"Memory {i}")
        
        self.assertEqual(len(self.memory.list_memories()), 3)
        self.memory.clear_memories()
        self.assertEqual(len(self.memory.list_memories()), 0)

if __name__ == '__main__':
    unittest.main()