import torch
import json
import os
from gliner import GLiNER
from transformers import AutoTokenizer

# Configuration
MODEL_NAME = "urchade/gliner_small-v2.1"
OUTPUT_DIR = "models/gliner"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"⬇️  Downloading GLiNER model: {MODEL_NAME}...")
try:
    model = GLiNER.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Warning: Standard load failed ({e}). Trying with checks disabled...")
    model = GLiNER.from_pretrained(MODEL_NAME, trust_remote_code=True)

model.eval()

# Save Config
with open(os.path.join(OUTPUT_DIR, "gliner_config.json"), "w") as f:
    json.dump(model.config.to_dict(), f, indent=2)

# Use base tokenizer
BASE_MODEL = "microsoft/deberta-v3-small" 
print(f"⬇️  Loading Tokenizer from base: {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- Wrappers ---

class EncoderWrapper(torch.nn.Module):
    def __init__(self, gliner_model):
        super().__init__()
        self.token_rep = gliner_model.model.token_rep_layer
        self.rnn = gliner_model.model.rnn
        
    def forward(self, input_ids, attention_mask):
        embedding = self.token_rep(input_ids, attention_mask)
        # Bypass 'pack_padded_sequence'
        output, _ = self.rnn.lstm(embedding)
        return output

class SpanRepWrapper(torch.nn.Module):
    def __init__(self, gliner_model):
        super().__init__()
        self.span_rep_layer = gliner_model.model.span_rep_layer

    def forward(self, embeddings, span_start, span_end):
        # NOTE: Assumes Batch Size = 1
        h_start = embeddings[0, span_start] 
        h_end = embeddings[0, span_end]
        
        # Inspection confirmed NO width embedding in this model version's SpanMarkerV0
        marker = self.span_rep_layer.span_rep_layer
        start_proj = marker.project_start(h_start)
        end_proj   = marker.project_end(h_end)
        
        # Concatenate: Start + End
        rep = torch.cat([start_proj, end_proj], dim=-1)
        
        # Final Projection
        out = marker.out_project(rep)
        return out

class PromptRepWrapper(torch.nn.Module):
    def __init__(self, gliner_model):
        super().__init__()
        self.prompt_rep = gliner_model.model.prompt_rep_layer

    def forward(self, embeddings):
        return self.prompt_rep(embeddings)

# --- EXPORT ---

# Dummy Data
B, L = 1, 512
input_ids = torch.randint(0, 1000, (B, L))
mask = torch.ones((B, L))

# 1. Export Encoder
print("⚙️  Exporting Encoder...")
encoder = EncoderWrapper(model)
torch.onnx.export(
    encoder,
    (input_ids, mask),
    os.path.join(OUTPUT_DIR, "gliner_encoder.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["embeddings"],
    opset_version=14
)

# Get output shape
with torch.no_grad():
    embs = encoder(input_ids, mask)
    HIDDEN_DIM = embs.shape[-1]

# 2. Export Span Rep
print("⚙️  Exporting Span Rep...")
span_rep = SpanRepWrapper(model)

# Dummy spans
span_start = torch.tensor([0, 1], dtype=torch.long)
span_end = torch.tensor([1, 3], dtype=torch.long)

torch.onnx.export(
    span_rep,
    (embs, span_start, span_end),
    os.path.join(OUTPUT_DIR, "gliner_span_rep.onnx"),
    input_names=["embeddings", "span_start", "span_end"],
    output_names=["span_embeddings"],
    dynamic_axes={
        "embeddings": {0: "batch", 1: "seq"},
        "span_start": {0: "num_spans"},
        "span_end": {0: "num_spans"},
        "span_embeddings": {0: "num_spans"}
    },
    opset_version=14
)

# 3. Export Prompt Rep
print("⚙️  Exporting Prompt Rep...")
prompt_rep = PromptRepWrapper(model)
dummy_label_input = torch.randn(3, HIDDEN_DIM) 

torch.onnx.export(
    prompt_rep,
    (dummy_label_input,),
    os.path.join(OUTPUT_DIR, "gliner_prompt_rep.onnx"),
    input_names=["label_embeddings"],
    output_names=["label_reps"],
    dynamic_axes={
        "label_embeddings": {0: "num_labels"},
        "label_reps": {0: "num_labels"}
    },
    opset_version=14
)

print(f"✅ Success! Models saved to {OUTPUT_DIR}")
