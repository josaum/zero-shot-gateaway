import onnxruntime as ort
import sys

def check_model(path):
    print(f"Checking {path}...")
    try:
        sess = ort.InferenceSession(path)
        print("✅ Load Success")
        for i in sess.get_inputs():
            print(f"Input: {i.name}, Shape: {i.shape}")
    except Exception as e:
        print(f"❌ Load Failed: {e}")

if __name__ == "__main__":
    check_model("models/picodet_layout.onnx")
    check_model("models/PP-DocLayout-L.onnx")
    check_model("models/layout.onnx")
