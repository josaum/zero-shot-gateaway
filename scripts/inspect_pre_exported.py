
from huggingface_hub import snapshot_download
import onnxruntime as ort
import os

def check_model():
    model_id = "juampahc/gliner_multi-v2.1-onnx"
    output_dir = "models/gliner_multi_v2.1"
    
    print(f"🚀 Downloading {model_id}...")
    snapshot_download(repo_id=model_id, local_dir=output_dir)
    
    print(f"📂 Model downloaded to {output_dir}")
    
    onnx_path = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        print("❌ model.onnx not found!")
        return

    print("🔍 Inspecting ONNX Signature...")
    session = ort.InferenceSession(onnx_path)
    
    print("\n📦 Inputs:")
    for i in session.get_inputs():
        print(f" - Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        
    print("\n📦 Outputs:")
    for o in session.get_outputs():
        print(f" - Name: {o.name}, Shape: {o.shape}, Type: {o.type}")

if __name__ == "__main__":
    check_model()
