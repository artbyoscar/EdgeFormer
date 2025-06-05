"""Utilities for analyzing model complexity and recommending compression strategies."""

from typing import Dict, List
import torch.nn as nn
import torch


class ModelComplexityAnalyzer:
    """Automatically analyze models for optimal compression."""

    def _calculate_sensitivity(self, param: torch.Tensor) -> float:
        """Return a simple sensitivity score based on parameter variance."""
        return float(torch.std(param.detach()))

    def analyze_sensitivity(self, model: nn.Module) -> Dict[str, float]:
        """Identify which parameters are most accuracy sensitive.

        Parameters
        ----------
        model: nn.Module
            The model to analyze.

        Returns
        -------
        Dict[str, float]
            Mapping of parameter name to sensitivity score.
        """
        sensitivity_map: Dict[str, float] = {}
        for name, param in model.named_parameters():
            sensitivity_map[name] = self._calculate_sensitivity(param)
        return sensitivity_map

    def recommend_compression_strategy(
        self, model: nn.Module, target_accuracy_loss: float = 1.0
    ) -> Dict[str, object]:
        """Recommend a compression configuration based on sensitivity analysis."""
        analysis = self.analyze_sensitivity(model)
        # Highest sensitivity layers are preserved in full precision
        sorted_layers = sorted(analysis.items(), key=lambda kv: kv[1], reverse=True)
        top_k = max(1, int(0.1 * len(sorted_layers)))
        skip_layers = [name for name, _ in sorted_layers[:top_k]]

        block_size = 64 if target_accuracy_loss < 0.5 else 128
        return {
            "skip_layers": skip_layers,
            "block_size": block_size,
            "symmetric": False,
        }
