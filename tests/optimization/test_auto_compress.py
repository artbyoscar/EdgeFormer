import unittest
import torch.nn as nn

from src.optimization.auto_compress import AutoCompressionSearch

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2,2)
    def forward(self,x):
        return self.l(x)

class TestAutoCompressionSearch(unittest.TestCase):
    def test_search_returns_none_without_impl(self):
        searcher = AutoCompressionSearch()
        model = DummyModel()
        # _bayesian_search is not implemented; expect AttributeError
        with self.assertRaises(AttributeError):
            searcher.search_optimal_configuration(model)

if __name__ == '__main__':
    unittest.main()
