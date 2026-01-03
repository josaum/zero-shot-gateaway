
import torch
import os
from gliner2 import GLiNER2

def export_to_onnx():
    output_dir = "models/gliner2-multi"
    model_name = "fastino/gliner2-multi-v1"
    
    print(f"🚀 Loading {model_name}...")
    model = GLiNER2.from_pretrained(model_name)
    model.eval()

    # Create dummy input
    text = ["This is a test sentence for export."]
    # The tokenizer call depends on the internal tokenizer wrapper
    # Inspect showed us it saves tokenizer.json, so it likely follows HF API
    # But for tracing, we need tensors.
    
    # We need to trace the underlying model forward pass.
    # GLiNER2 typically has a .predict() or .extract() method that does pre-processing.
    # We want to export the neural network part (Encoder + Heads).
    
    # Let's inspect the forward signature of the underlying torch module
    # Usually model.model or model.encoder
    
    # Inspect forward signature to know what to pass
    import inspect
    print(f"🔍 Forward Signature: {inspect.signature(model.forward)}")
    
    # Load tokenizer explicitly since it might not be attached
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Based on signature, we prepare args. 
    # v2 usually needs input_ids, attention_mask, and maybe more.
    # We will assume standard HF args first, but if forward has required args like 'words', we need to provide them.
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Dictionary of inputs
    model_inputs = (input_ids, attention_mask)
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    }
    
    # If the signature has other required args, we will see it in the logs and fix.
    
    onnx_path = os.path.join(output_dir, "gliner2_multi.onnx")
    print(f"📦 Exporting entire model to {onnx_path}...")
    
    try:
        torch.onnx.export(
            model, # Export the root model
            model_inputs,
            onnx_path,
            input_names=input_names,
            output_names=["logits", "span_logits"], # Guessing
            dynamic_axes=dynamic_axes,
            opset_version=14
        )
        print("✅ ONNX Export Successful!")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        # If it failed due to missing args, we will know from the signature printed above.

if __name__ == "__main__":
    export_to_onnx()
