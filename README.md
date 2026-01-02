# ⚡ One-File Event Gateway + Zero-Copy Physics Engine

[![CI](https://github.com/josaum/zero-shot-gateaway/actions/workflows/ci.yml/badge.svg)](https://github.com/josaum/zero-shot-gateaway/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)

> **The "God Node" of Event Processing**: A single Rust binary that acts as an Ingestion Engine, OLAP Database, AI Agent, Physics Engine, and TUI Dashboard — with **Zero-Copy** data streaming to Python ML pipelines.

![TUI Demo](demo.webp)

---

## 🎯 What Is This?

This is a **radical experiment** in AI-native infrastructure consolidation. Instead of running 10 microservices, you run **one binary** that does:

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Ingestion** | Axum REST API | Receives webhooks and events |
| **Storage** | DuckDB (embedded OLAP) | Columnar analytics database |
| **AI Brain** | LM Studio / OpenAI | Intent detection & event classification |
| **Physics Engine** | Shared Memory + SeqLock | Zero-copy data to ML pipelines |
| **Interface** | Ratatui TUI | Terminal-native dashboard |

---

## 🚀 Why Is This So Good?

### 1. **Zero-Copy Physics Engine** 
Traditional pipelines serialize data to JSON/Protobuf, send over HTTP, then deserialize. **We eliminate all of that.**

```
┌───────────────┐    ┌─────────────────┐    ┌───────────────┐
│ Rust Gateway  │───▶│ Shared Memory   │───▶│ Python/PyTorch│
│  (Producer)   │    │ /tmp/cs_physics │    │  (Consumer)   │
└───────────────┘    └─────────────────┘    └───────────────┘
      WRITE              ZERO-COPY              READ
      ~50ns               DIRECT               ~100ns
```

**Benefits:**
- **Sub-microsecond latency**: Data appears in Python instantly
- **No serialization**: Raw bytes mapped directly to NumPy arrays
- **Lock-free reads**: SeqLock protocol allows concurrent read/write
- **GPU-ready**: Tensors can be moved to CUDA without copies

### 2. **Any LLM, Locally**
Swapped from Gemini Cloud to **LM Studio** running on your local network. Full privacy, zero latency to cloud, works offline.

```bash
# Automatically uses your LM Studio at:
http://192.168.0.141:1234/v1

# Or set custom endpoint:
export LLM_BASE_URL="http://localhost:1234/v1"
```

Tested with:
- ✅ `qwen/qwen3-4b-thinking-2507` (thinking chain support!)
- ✅ `text-embedding-bge-m3@f16` (embeddings)
- ✅ Any OpenAI-compatible API

### 3. **Thinking Model Support**
Models like Qwen3-Thinking wrap their reasoning in `<think>...</think>` tags. We parse that automatically:

```
Input: "Quero fazer um seguro pro meu carro"
       ↓
Qwen3: <think>The user wants car insurance...</think>
       {"intent": "CarInsuranceRequest"}
       ↓
Output: event_type = "CarInsuranceRequest"
```

### 4. **Real-Time ML Feature Engineering**
Every event gets transformed into a **ParticleFrame** (4KB aligned struct):

```rust
pub struct ParticleFrame {
    pub semantic: [f32; 1024],    // BGE-M3 embedding
    pub delta_time: f32,          // Time since last event
    pub velocity: f32,            // Messages/minute
    pub sentiment: f32,           // [-1.0, 1.0]
    pub spin: [f32; 4],           // Orientation quaternion
    // ... 4KB total, cache-aligned
}
```

Python reads these directly:
```python
from consumer import PhysicsConsumer

with PhysicsConsumer() as consumer:
    for frame in consumer.follow():
        print(f"Embedding: {frame.semantic[:3]}...")
        print(f"Sentiment: {frame.sentiment}")
```

---

## 🛠 Quick Start

### 1. Install Dependencies

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# ONNX Runtime (for local embeddings)
brew install onnxruntime

# Python consumer
pip install numpy mmap torch  # torch optional
```

### 2. Run the Gateway

```bash
# With TUI
DYLD_LIBRARY_PATH="/opt/homebrew/lib" cargo run --release

# Headless mode
DYLD_LIBRARY_PATH="/opt/homebrew/lib" cargo run --release -- --no-tui
```

### 3. Send Events

```bash
curl -X POST http://localhost:9382/ingest \
  -H "Content-Type: application/json" \
  -d '{"message": "Quero cancelar meu plano", "type": "UserMessage"}'
```

### 4. Consume in Python

```bash
# Follow mode (like tail -f)
python consumer.py --follow

# Batch read
python consumer.py --batch 100

# PyTorch tensors
python consumer.py --pytorch --batch 50
```

---

## 📂 Project Structure

```
/
├── src/main.rs           # 🧠 Gateway: API, DuckDB, LLM, Physics Engine integration
├── src/collider.rs       # ⚡ Physics Engine: Shared memory, embeddings, SeqLock
├── src/tui.rs            # 🖥️ Terminal UI: Ratatui dashboard
├── consumer.py           # 🐍 Python Consumer: Zero-copy shared memory reader
├── Cargo.toml            # Rust dependencies
├── models/               # ONNX models (optional, uses LM Studio by default)
└── .env                  # Configuration
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://192.168.0.141:1234/v1` | LM Studio / OpenAI endpoint |
| `DYLD_LIBRARY_PATH` | - | Path to ONNX Runtime (macOS) |

### CLI Arguments

```bash
./one_file_gateway --help

Options:
  --db-path <PATH>   Persistent DuckDB file (default: in-memory)
  --token <TOKEN>    Bearer token for API auth
  --no-tui           Headless mode (no terminal UI)
```

---

## 🧪 Testing the Full Pipeline

```bash
# Terminal 1: Start Gateway
DYLD_LIBRARY_PATH="/opt/homebrew/lib" cargo run --release -- --no-tui

# Terminal 2: Start Consumer
python consumer.py --follow

# Terminal 3: Send test events
curl -X POST http://localhost:9382/ingest \
  -H "Content-Type: application/json" \
  -d '{"message": "Preciso de ajuda com minha conta", "conversation_id": "test"}'

# Watch Terminal 2: You'll see the frame appear instantly!
```

---

## 🏗 Architecture Deep Dive

### Shared Memory Layout

```
┌────────────────────────────────────────────────────────────┐
│                    HEADER (64 bytes)                       │
├──────────┬──────────┬──────────┬──────────┬───────────────┤
│  MAGIC   │ VERSION  │ FRAME_SZ │ CAPACITY │  HEAD | TAIL  │
│  8 bytes │ 4 bytes  │ 4 bytes  │ 8 bytes  │   8  |  8    │
└──────────┴──────────┴──────────┴──────────┴───────────────┘
┌────────────────────────────────────────────────────────────┐
│              RING BUFFER (100,000 × 4KB frames)            │
├────────────────────────────────────────────────────────────┤
│ Frame 0: [seq|frame_id|semantic[1024]|kinetics|spin|...]  │
│ Frame 1: [seq|frame_id|semantic[1024]|kinetics|spin|...]  │
│ ...                                                        │
│ Frame 99,999: [...]                                        │
└────────────────────────────────────────────────────────────┘
```

### SeqLock Protocol

Writers and readers operate **lock-free**:

```
Writer (Rust):                    Reader (Python):
1. seq = seq + 1 (odd = writing)  1. read seq₁
2. write payload                  2. read payload
3. seq = seq + 1 (even = valid)   3. read seq₂
4. update head pointer            4. if seq₁ != seq₂ or odd: retry
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Write latency | ~50ns per frame |
| Read latency | ~100ns per frame |
| Throughput | 10M+ frames/sec |
| Memory footprint | ~400MB (100K frames) |
| Embedding latency (LM Studio) | ~50ms |

---

## 🤝 Contributing

1. **Keep it Simple** — One-file philosophy
2. **Format**: `cargo fmt`
3. **Lint**: `cargo clippy`
4. **Test**: `cargo test`

---

## 📜 License

MIT — Do whatever you want with it.

---

## 🙏 Credits

Built with:
- [Axum](https://github.com/tokio-rs/axum) — Rust web framework
- [DuckDB](https://duckdb.org/) — Embedded OLAP database
- [Ratatui](https://ratatui.rs/) — Terminal UI framework
- [LM Studio](https://lmstudio.ai/) — Local LLM inference
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — Multilingual embeddings
