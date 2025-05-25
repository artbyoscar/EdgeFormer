# Create: src/optimization/auto_compress.py
class AutoCompressionSearch:
    """Automatically find optimal compression settings"""
    
    def search_optimal_configuration(self, model, target_accuracy_loss=1.0):
        """Use grid search / Bayesian optimization to find best settings"""
        search_space = {
            "block_size": [32, 64, 128],
            "symmetric": [True, False],
            "skip_layer_patterns": [
                ["embeddings", "head"],
                ["embeddings"],
                ["head"],
                []
            ],
            "calibration_percentile": [0.99, 0.995, 0.999]
        }
        
        best_config = self._bayesian_search(model, search_space, target_accuracy_loss)
        return best_config