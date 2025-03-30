"""
Memory adapter for EdgeFormer healthcare demo.

This adapter bridges compatibility between the MemoryRetriever and HTPSMemory,
providing any missing methods required by the demo.
"""

import torch
import logging
from src.model.associative_memory.htps_memory import HTPSMemory

logger = logging.getLogger(__name__)

class HTPSMemoryAdapter:
    """Adapter for HTPSMemory to provide additional methods needed for the demo."""
    
    def __init__(self, memory):
        """
        Initialize adapter for HTPSMemory.
        
        Args:
            memory: HTPSMemory instance to adapt
        """
        self.memory = memory
    
    def is_empty(self):
        """Check if memory is empty."""
        if not hasattr(self.memory, 'vectors') or self.memory.vectors is None:
            return True
        return len(self.memory.vectors) == 0
    
    def get_vectors(self):
        """Get memory vectors."""
        if hasattr(self.memory, 'get_vectors'):
            return self.memory.get_vectors()
        
        # Fallback if get_vectors is not available
        if hasattr(self.memory, 'vectors'):
            return self.memory.vectors
        
        # Return empty tensor if no vectors found
        logger.warning("No vectors found in memory, returning empty tensor")
        return torch.tensor([])
    
    def get_texts(self):
        """Get memory text descriptions."""
        if hasattr(self.memory, 'get_texts'):
            return self.memory.get_texts()
        
        # Fallback if get_texts is not available
        if hasattr(self.memory, 'texts'):
            return self.memory.texts
        
        # Return empty list if no texts found
        logger.warning("No texts found in memory, returning empty list")
        return []
    
    def add_entry(self, vector, text):
        """Add entry to memory."""
        if hasattr(self.memory, 'add_entry'):
            return self.memory.add_entry(vector, text)
        
        # Implement fallback if needed
        logger.warning("add_entry not available in memory, entry not added")
    
    def __getattr__(self, name):
        """Delegate other method calls to the wrapped memory."""
        return getattr(self.memory, name)