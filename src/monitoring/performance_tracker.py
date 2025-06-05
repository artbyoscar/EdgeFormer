"""Runtime monitoring of model predictions."""

from typing import List, Dict
import time
import torch
import math


class DriftDetector:
    """Simple drift detector based on confidence statistics."""

    def __init__(self, window: int = 50, threshold: float = 0.1) -> None:
        self.window = window
        self.threshold = threshold
        self.history: List[float] = []

    def detect_drift(self, metrics: Dict[str, float]) -> bool:
        self.history.append(metrics["confidence"])
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < self.window:
            return False
        mean_conf = sum(self.history) / len(self.history)
        return abs(metrics["confidence"] - mean_conf) > self.threshold


class PerformanceTracker:
    """Track inference metrics and detect performance issues."""

    def __init__(self) -> None:
        self.metrics_history: List[Dict[str, float]] = []
        self.drift_detector = DriftDetector()

    def _calculate_entropy(self, prediction: torch.Tensor) -> float:
        probs = torch.softmax(prediction.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return float(entropy)

    def _analyze_input(self, input_data: torch.Tensor) -> Dict[str, float]:
        return {"mean": float(input_data.mean()), "std": float(input_data.std())}

    def _trigger_recompression_alert(self) -> None:
        print("WARNING: potential accuracy drift detected")

    def track_inference(self, input_data: torch.Tensor, prediction: torch.Tensor, confidence: float) -> Dict[str, float]:
        """Monitor an inference call for drift."""
        metrics = {
            "timestamp": time.time(),
            "confidence": float(confidence),
            "prediction_entropy": self._calculate_entropy(prediction),
            "input_characteristics": self._analyze_input(input_data),
        }
        if self.drift_detector.detect_drift(metrics):
            self._trigger_recompression_alert()
        self.metrics_history.append(metrics)
        return metrics
