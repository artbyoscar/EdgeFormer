# Create: docs/api_reference.py
"""
EdgeFormer API Reference

Quick Start:
>>> from edgeformer import quantize_model
>>> compressed = quantize_model(your_model, mode="medical_grade")
>>> # Guaranteed <0.5% accuracy loss for medical applications

Production Deployment:
>>> config = EdgeFormerConfig.from_preset("automotive_adas")
>>> compressed = quantize_model(your_model, config=config)
>>> # Safety-certified accuracy for automotive deployment

Advanced Usage:
>>> optimizer = AutoCompressionSearch()
>>> optimal_config = optimizer.search_optimal_configuration(
...     your_model, 
...     target_accuracy_loss=0.3
... )
>>> compressed = quantize_model(your_model, config=optimal_config)
"""