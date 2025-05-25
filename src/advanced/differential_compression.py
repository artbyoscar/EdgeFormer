# Create: src/advanced/differential_compression.py
class DifferentialCompressor:
    """Compress model updates instead of full models"""
    
    def compress_model_update(self, base_model, fine_tuned_model):
        """Compress only the differences between models"""
        delta = self._compute_model_delta(base_model, fine_tuned_model)
        compressed_delta = self._compress_delta(delta)
        
        # This allows for tiny update packages instead of full model shipping
        return {
            "base_model_id": base_model.id,
            "compressed_delta": compressed_delta,
            "compression_ratio": self._calculate_delta_compression(delta, compressed_delta)
        }