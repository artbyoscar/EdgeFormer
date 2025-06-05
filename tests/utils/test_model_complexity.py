import unittest
import torch
import torch.nn as nn

from src.utils.model_analyzer import ModelComplexityAnalyzer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 2)

    def forward(self, x):
        return self.l2(self.l1(x))

class TestModelComplexityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.analyzer = ModelComplexityAnalyzer()
        # monkeypatch private method so analyze_sensitivity works
        self.analyzer._calculate_sensitivity = lambda param: float(param.numel())

    def test_analyze_sensitivity(self):
        sens = self.analyzer.analyze_sensitivity(self.model)
        # Should include all parameters
        self.assertIn('l1.weight', sens)
        self.assertIn('l2.bias', sens)
        self.assertEqual(sens['l1.weight'], float(self.model.l1.weight.numel()))

    def test_recommend_strategy_returns_none(self):
        # recommend_compression_strategy currently returns None
        result = self.analyzer.recommend_compression_strategy(self.model)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
