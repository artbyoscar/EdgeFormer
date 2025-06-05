import unittest
import torch

from src.utils.quantization import AdaptiveInt4Quantizer
from src.utils.model_analyzer import ModelComplexityAnalyzer
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator
from src.optimization.auto_compress import AutoCompressionSearch
from src.monitoring.performance_tracker import PerformanceTracker
from src.model.bert_edgeformer import BERTEdgeFormer
from src.model.edgeformer import EdgeFormerConfig


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class RoadmapModuleTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()

    def test_adaptive_quantizer(self):
        q = AdaptiveInt4Quantizer()
        quant = q.apply_to_model(self.model)
        x = torch.randn(1, 4)
        out = quant(x)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(q.quantized_layers)

    def test_model_analyzer(self):
        analyzer = ModelComplexityAnalyzer()
        strategy = analyzer.recommend_compression_strategy(self.model)
        self.assertIn("skip_layers", strategy)

    def test_comprehensive_evaluator(self):
        evaluator = ComprehensiveEvaluator()
        inp = torch.randn(1, 4)
        metrics = evaluator.evaluate_compressed_model(self.model, self.model, inp)
        self.assertIn("accuracy_loss", metrics)

    def test_auto_compression_search(self):
        search = AutoCompressionSearch()
        cfg = EdgeFormerConfig(hidden_size=4, num_hidden_layers=1, num_attention_heads=1, intermediate_size=4, latent_size_factor=2)
        model = BERTEdgeFormer(cfg)
        result = search.search_optimal_configuration(model, target_accuracy_loss=10.0)
        self.assertIsNotNone(result)

    def test_performance_tracker(self):
        tracker = PerformanceTracker()
        data = torch.randn(1, 4)
        pred = torch.randn(1, 2)
        tracker.track_inference(data, pred, 0.5)
        self.assertEqual(len(tracker.metrics_history), 1)


if __name__ == "__main__":
    unittest.main()
