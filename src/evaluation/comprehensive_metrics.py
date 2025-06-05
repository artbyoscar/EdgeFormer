"""Comprehensive evaluation utilities for compressed models."""

from typing import Dict
import time
import torch
import numpy as np


class ComprehensiveEvaluator:
    """Evaluate a compressed model using multiple metrics."""

    def evaluate_compressed_model(self, original, compressed, test_data) -> Dict[str, float]:
        """Run evaluation comparing an original and a compressed model."""
        original.eval()
        compressed.eval()
        with torch.no_grad():
            orig_out = original(test_data)
            comp_out = compressed(test_data)

        metrics = {
            "accuracy_loss": self._accuracy_loss(orig_out, comp_out),
            "per_class_accuracy": self._per_class_analysis(orig_out, comp_out),
            "confidence_distribution": self._confidence_analysis(comp_out),
            "inference_speedup": self._measure_speedup(original, compressed, test_data),
            "memory_efficiency": self._memory_analysis(original, compressed),
            "numerical_stability": self._stability_analysis(comp_out),
            "adversarial_robustness": self._adversarial_test(compressed, test_data),
            "out_of_distribution": self._ood_performance(compressed, test_data),
            "calibration_quality": self._calibration_analysis(comp_out),
        }
        return metrics

    def _accuracy_loss(self, original_output, compressed_output) -> float:
        """Compute simple mean squared error between two outputs."""
        orig = torch.as_tensor(original_output).float().flatten()
        comp = torch.as_tensor(compressed_output).float().flatten()
        return float(torch.mean((orig - comp) ** 2))

    def _per_class_analysis(self, original_output, compressed_output):
        """Dummy per-class accuracy placeholder."""
        return self._accuracy_loss(original_output, compressed_output)

    def _confidence_analysis(self, output):
        """Return average softmax confidence."""
        probs = torch.softmax(torch.as_tensor(output).float(), dim=-1)
        return float(probs.max(dim=-1).values.mean())

    def _measure_speedup(self, original, compressed, data):
        """Measure simple inference speedup."""
        def _timed(model):
            start = time.time()
            with torch.no_grad():
                model(data)
            return time.time() - start

        o = _timed(original)
        c = _timed(compressed)
        return o / c if c > 0 else 1.0

    def _memory_analysis(self, original, compressed):
        """Estimate memory efficiency based on parameter count."""
        orig_params = sum(p.numel() for p in original.parameters())
        comp_params = sum(p.numel() for p in compressed.parameters())
        if comp_params == 0:
            return 1.0
        return orig_params / comp_params

    def _stability_analysis(self, output):
        """Check output variance as a stability proxy."""
        return float(torch.var(torch.as_tensor(output)))

    def _adversarial_test(self, model, data):
        """Placeholder adversarial robustness score."""
        with torch.no_grad():
            model(data)
        return 1.0  # Stub score

    def _ood_performance(self, model, data):
        """Placeholder out-of-distribution metric."""
        with torch.no_grad():
            model(data)
        return 1.0

    def _calibration_analysis(self, output):
        """Simple calibration metric based on confidence entropy."""
        probs = torch.softmax(torch.as_tensor(output).float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return float(entropy.mean())
