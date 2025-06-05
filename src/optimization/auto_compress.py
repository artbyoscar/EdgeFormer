"""Automatic search for compression hyperparameters."""

from typing import Dict, Any, Iterable
from itertools import product
import torch
from src.utils.quantization import Int4Quantizer


class AutoCompressionSearch:
    """Simple grid search for optimal INT4 quantization parameters."""

    def _evaluate(self, model, quantizer: Int4Quantizer, sample_input) -> float:
        """Return mean squared error between original and quantized model outputs."""
        q_model = quantizer.apply_to_model(model)
        with torch.no_grad():
            orig = model(sample_input)
            comp = q_model(sample_input)

        if isinstance(orig, (list, tuple)):
            orig = orig[0]
        if isinstance(comp, (list, tuple)):
            comp = comp[0]
        if isinstance(orig, dict):
            orig = next(iter(orig.values()))
        if isinstance(comp, dict):
            comp = next(iter(comp.values()))

        orig_t = torch.as_tensor(orig).float().flatten()
        comp_t = torch.as_tensor(comp).float().flatten()
        return float(torch.mean((orig_t - comp_t) ** 2))

    def _grid_search(self, model, space: Dict[str, Iterable], target_loss: float):
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            sample_input = torch.randint(0, model.config.vocab_size, (1, 2))
        else:
            sample_input = torch.randn(1, getattr(model.config, "hidden_size", 10))
        best_cfg = None
        best_loss = float("inf")
        for block, sym in product(space["block_size"], space["symmetric"]):
            quantizer = Int4Quantizer(block_size=block, symmetric=sym)
            loss = self._evaluate(model, quantizer, sample_input)
            if loss < best_loss and loss <= target_loss:
                best_loss = loss
                best_cfg = {"block_size": block, "symmetric": sym}
        return best_cfg

    def search_optimal_configuration(self, model, target_accuracy_loss: float = 1.0) -> Dict[str, Any]:
        """Search the space of quantization parameters and return the best config."""
        search_space = {
            "block_size": [32, 64, 128],
            "symmetric": [True, False],
        }
        return self._grid_search(model, search_space, target_accuracy_loss)
