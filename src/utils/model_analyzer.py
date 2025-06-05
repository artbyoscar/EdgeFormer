# Create: src/utils/model_analyzer.py
class ModelComplexityAnalyzer:
    """Automatically analyze model to recommend optimal compression strategy"""
    
    def analyze_sensitivity(self, model):
        """Identify which layers are most accuracy-sensitive"""
        sensitivity_map = {}
        for name, param in model.named_parameters():
            # Analyze gradient magnitude, parameter variance, etc.
            sensitivity_score = self._calculate_sensitivity(param)
            sensitivity_map[name] = sensitivity_score
        return sensitivity_map
    
    def recommend_compression_strategy(self, model, target_accuracy_loss=1.0):
        """AI-powered recommendation for compression settings"""
        analysis = self.analyze_sensitivity(model)
        # Return optimized configuration based on model characteristics
