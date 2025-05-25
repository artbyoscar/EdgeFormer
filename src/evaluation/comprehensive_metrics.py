# Create: src/evaluation/comprehensive_metrics.py
class ComprehensiveEvaluator:
    """Advanced evaluation beyond accuracy"""
    
    def evaluate_compressed_model(self, original, compressed, test_data):
        """Comprehensive model evaluation"""
        metrics = {
            # Accuracy metrics
            "accuracy_loss": self._accuracy_loss(original, compressed, test_data),
            "per_class_accuracy": self._per_class_analysis(original, compressed, test_data),
            "confidence_distribution": self._confidence_analysis(original, compressed, test_data),
            
            # Performance metrics  
            "inference_speedup": self._measure_speedup(original, compressed),
            "memory_efficiency": self._memory_analysis(original, compressed),
            "numerical_stability": self._stability_analysis(compressed),
            
            # Robustness metrics
            "adversarial_robustness": self._adversarial_test(compressed),
            "out_of_distribution": self._ood_performance(compressed, test_data),
            "calibration_quality": self._calibration_analysis(compressed, test_data)
        }
        return metrics