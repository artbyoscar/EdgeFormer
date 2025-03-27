# Create a file named test_onnx_directml.py
import onnxruntime as ort
import numpy as np

# Check available providers
providers = ort.get_available_providers()
print(f"Available ONNX Runtime providers: {providers}")

# Check if DirectML is available
if 'DmlExecutionProvider' in providers:
    print("DirectML is available for acceleration!")
    
    # Create a simple session with DirectML provider
    options = ort.SessionOptions()
    session = ort.InferenceSession("path_to_model.onnx", 
                                   providers=['DmlExecutionProvider'],
                                   sess_options=options)
    print("Successfully created DirectML session")
else:
    print("DirectML provider not found in available providers.")