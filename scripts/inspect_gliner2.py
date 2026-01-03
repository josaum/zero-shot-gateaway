
import os
import torch
import json
from gliner2 import GLiNER2

def export_gliner2():
    print("🚀 Downloading GLiNER2 Multi...")
    try:
        model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1")
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    output_dir = "models/gliner2-multi"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Exporting to {output_dir}...")

    print(f"Model attributes: {list(model.__dict__.keys())}")
    
    # helper to find tokenizer
    tokenizer = None
    if hasattr(model, 'tokenizer'): tokenizer = model.tokenizer
    elif hasattr(model, 'model') and hasattr(model.model, 'tokenizer'): tokenizer = model.model.tokenizer
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'tokenizer'): tokenizer = model.encoder.tokenizer
    
    if tokenizer:
        print("✅ Found tokenizer!")
        tokenizer.save_pretrained(output_dir)
    else:
        print("⚠️ Tokenizer not found in standard locations.")
        # Print sub-attributes of key components
        for key in ['encoder', 'model', 'backbone']:
            if hasattr(model, key):
                print(f"Attributes of {key}: {dir(getattr(model, key))}")

    return # Stop here to avoid crash on mock input
    
    # We need to understand the forward signature of gliner2 to export correctly.
    # Typically v2 might be a single model or still split.
    # Let's try to export the underlying encoder if it's the main component.
    
    # According to research, GLiNER2 might use a unified architecture.
    # Let's print the model structure to see what we are dealing with.
    print(model)

    # Attempt to trace the 'forward' method if possible, or key subcomponents.
    # For now, let's just inspect the model structure to decide on export strategy.
    
    # Inspect configuration
    config = model.config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print("✅ Config saved")

if __name__ == "__main__":
    try:
        import gliner2
        export_gliner2()
    except ImportError:
        print("❌ 'gliner2' package not found. Please install it with: pip install gliner2")
