import onnxruntime as ort
import sys

model_path = "models/gliner_multi_v2.1/model.onnx"

try:
    sess = ort.InferenceSession(model_path)
    print(f"Model loaded: {model_path}")
    
    print("\nInputs:")
    for inp in sess.get_inputs():
        print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
        
    print("\nOutputs:")
    for out in sess.get_outputs():
        print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")
        
except Exception as e:
    print(f"Error: {e}")
