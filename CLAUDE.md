# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build and run with Full Capability (OCR, NER, etc.)
DYLD_LIBRARY_PATH="/opt/homebrew/lib" cargo run --release --features oar

# Headless mode (no TUI)
DYLD_LIBRARY_PATH="/opt/homebrew/lib" cargo run --release --features oar -- --no-tui

# Python consumer
python scripts/consumer.py --follow    # Real-time monitoring
python scripts/consumer.py --batch 100 # Batch processing
python scripts/consumer.py --pytorch   # PyTorch tensor output

# Standard Rust commands
cargo check --features oar  # Type checking
cargo clippy --features oar # Lint
cargo fmt                   # Format
```

## Architecture Overview

**Single-binary gateway** combining:

| Component | File | Technology |
|-----------|------|------------|
| API Server | `main.rs` | Axum (port 9382) |
| Database | `main.rs` | DuckDB + Arrow IPC |
| Physics Engine | `collider.rs` | Shared Memory + SeqLock |
| NER Engine | `gliner.rs` | Local ONNX (GLiNER) |
| OCR Pipeline | `ocr_pipeline.rs` | Local ONNX (OAR-OCR) |
| LLM Integration | `main.rs` | LM Studio / OpenAI API |
| Terminal UI | `tui.rs` | Ratatui |
| Python Consumer | `scripts/consumer.py` | mmap + numpy |

## Key Modules

### `src/collider.rs` — Physics Engine
- **ParticleFrame**: 4KB aligned struct with semantic embeddings, kinetics, spin
- **SeqLock Protocol**: Lock-free concurrent read/write
- **Deep Zero-Copy**: Uses `ort::IoBinding` to write ONNX outputs directly to shared memory
- **Shared Memory**: `/tmp/cs_physics` (macOS) or `/dev/shm/cs_physics` (Linux)

### `src/gliner.rs` — NER Engine
- **Zero-Shot NER**: Running `gliner_base.onnx` locally
- **Output**: Extracts Entities and spans for structured storage

### `src/ocr_pipeline.rs` — Document AI
- **OAR-OCR**: Detection, Recognition, Layout, Table Extraction
- **Pipelines**: `legacy`, `oar-structure`, `oar-vl`

### `src/main.rs` — Gateway Core
- **LLM Integration**: `call_llm()` calls LM Studio with thinking-model support
- **strip_thinking_chain()**: Extracts JSON from `<think>...</think>` output
- **ingest_handler()**: Processes events → LLM → GLiNER → Physics Engine → DuckDB

### `scripts/consumer.py` — Python Consumer
- **PhysicsConsumer**: Memory-mapped reader with SeqLock
- **PyTorchConsumer**: Returns `torch.Tensor` directly
- **Modes**: `--follow`, `--batch N`, `--pytorch`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://192.168.0.141:1234/v1` | LM Studio endpoint |
| `DYLD_LIBRARY_PATH` | — | ONNX Runtime path (macOS) |

## Critical Constraints

1. **Zero-Copy Philosophy**: Data flows without serialization
   - DuckDB → Arrow RecordBatch → TUI (no structs)
   - Physics Engine → Shared Memory → Python numpy (no copies)

2. **Lock-Free Reads**: SeqLock allows Python to read while Rust writes

3. **Thinking Model Support**: Parser handles `<think>` tags automatically

## Data Flow

```
POST /ingest
     │
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ LLM Studio  │───▶│  DuckDB     │───▶│ tx_notify   │
│ (intent)    │    │ (persist)   │    │ (broadcast) │
└─────────────┘    └─────────────┘    └─────────────┘
     │                                       │
     ▼                                       ▼
┌─────────────┐                      ┌─────────────┐
│  Collider   │                      │    TUI      │
│ (smash)     │                      │  (refresh)  │
└─────────────┘                      └─────────────┘
     │
     ▼
┌─────────────┐    ┌─────────────┐
│ Shared Mem  │───▶│   Python    │
│ /tmp/...    │    │  consumer   │
└─────────────┘    └─────────────┘
```

## Code Style

- `snake_case` for variables/functions
- Keep `main.rs` as the orchestration layer
- Heavy logic goes into dedicated modules (`collider.rs`, `tui.rs`)
