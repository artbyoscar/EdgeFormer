import torch

def quantize_model(model, quantization="int8"):
    """Apply quantization to model weights."""
    if quantization == "int8":
        # Simple dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    elif quantization == "int4":
        # For int4, you'd need a custom implementation
        # This is a placeholder for a more complex implementation
        return model
    else:
        return model