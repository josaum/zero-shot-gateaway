# ⚡ One-File Event Gateway (TUI Edition)

[![CI](https://github.com/josaum/zero-shot-gateaway/actions/workflows/ci.yml/badge.svg)](https://github.com/josaum/zero-shot-gateaway/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)

> **The "God Node" of Event Processing**: A single Rust binary that acts as an Ingestion Engine, OLAP Database, AI Agent, and TUI Dashboard.

![TUI Demo](demo.webp)

This project pushes the boundaries of the **"One-File"** architecture. It is a highly capable, agentic data system contained entirely within a single `main.rs`, integrating:

*   **TUI (Ratatui)**: Zero-latency, keyboard-first terminal interface.
*   **DuckDB + Arrow IPC**: **Zero-copy** data streaming from OLAP to UI.
*   **Real-time Broadcast**: Native `tokio::sync::broadcast` signaling (no polling).
*   **Gemini AI**: Embedded chat agent for intent detection and workflow automation.

---

## 💡 Why "One File"?

In a world of microservices and complex Kubernetes manifests, **Simplicity is Power**. 
This project demonstrates that you can build a production-grade, concurrent, and stateful application without the sprawl. It serves as:
1.  **A provocative study** in Rust state management (`Arc`, `Mutex`, `RwLock`).
2.  **The ultimate deployment unit**: Just copy the binary and run. No containers required.

---

## 🎯 Use Cases

### 🛠 The Data Engineer's Terminal
Stop spinning up UI tools to debug webhooks.
*   **Run this Gateway locally.**
*   **Ingest events** via `curl`.
*   **Visualize instantly** in the terminal with Arrow-speed rendering.

### 🧠 The AI Agent's "Cortex"
Agents need memory and interfaces.
*   Send your agent's logs or observations to the gateway.
*   The gateway **learns the structure** of the memories.
*   **Chat with your data** natively in the TUI using the embedded AI assistant.

### ⚡ Arrow Zero-Copy Performance
Demonstrating the raw power of modern Rust data stacks.
*   DuckDB stores the data.
*   Data is exported directly as **Arrow RecordBatches**.
*   The TUI renders directly from columnar arrays. **Zero deserialization overhead.**

---

## 🚀 Features

1.  **Arrow IPC Zero-Copy**: Data flows from DuckDB to the screen without deserializing into Rust structs.
2.  **Real-time Broadcasting**: Ingestion triggers instant UI updates via internal signaling. No HTTP polling.
3.  **Learned Schemas**: Automatically infers types from JSON events.
4.  **AI Chat Interface**: Context-aware agent for querying data and triggering workflows.
5.  **Reflexive Configuration**: Reconfigures itself based on `SystemConfig` events.

---

## 🛠 Prerequisites

### 1. Rust
Ensure you have the latest stable Rust installed.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Gemini API Key
Required for the AI Chat features.
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## 🏃‍♂️ Running the Gateway

Because this uses a single binary, you just need to run:

```bash
# Optimized release build is recommended for TUI performance
cargo run --release
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| `1-4` | Switch Tabs (Events, Schemas, Chat, Metrics) |
| `j` / `k` | Scroll Lists |
| `e` | Enter Chat/Edit Mode (Chat Tab) |
| `Esc` | Exit Edit Mode |
| `Enter` | Send Message |
| `q` | Quit |

---

## 📂 Project Structure

```
/
├── src/main.rs           # 🧠 THE BRAIN: logic, db, api. logic in modules/tui.rs
├── src/tui.rs            # 🖥️ THE FACE: Ratatui implementation
├── Cargo.toml            # Dependencies
├── AGENTS.md             # Developer guide
├── README.md             # This file
├── llms.txt              # AI Context file
└── .env                  # Secrets
```

> [!IMPORTANT]
> **One-File Philosophy**: While we split `tui.rs` for visual sanity, the core "Gateway" logic remains centralized. The TUI is treated as a "view" layer.

---

## 📖 Usage Examples

### 1. Ingestion (The Learning Phase)

**Ingest a `UserSignup` event:**
```bash
curl -X POST http://127.0.0.1:9382/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "type": "UserSignup",
    "email": "alice@example.com",
    "region": "US-East",
    "plan": "Pro"
  }'
```

*Check the TUI: You will see the event appear **instantly** in the "Events" tab.*

---

### 2. The AI Chat (The Acting Phase)

1. Switch to **Chat** tab (Press `3`).
2. Press `e` to enter edit mode.
3. Type: *"I want to check recent signups."*
4. Press `Enter`.
5. The AI (Gemini) will analyze the learned schemas and respond.

---

## 📊 Monitoring

### Metrics
The **Metrics** tab (Press `4`) visualizes real-time throughput and schema learning counts using ASCII bar charts.

---

## 🧠 Architecture Notes

- **Zero-Copy**: Uses `duckdb::vtab_arrow` to stream `RecordMatch`es.
- **Signal**: `tokio::sync::broadcast` for event propagation.
- **Concurrency**: State shared via `Arc<RwLock<AppState>>`.
- **Storage**: In-memory DuckDB (default) or fast local disk.

---

## 👥 Contributing

1. **Keep it Simple**.
2. **Format Code**: `cargo fmt`
3. **Check Clippy**: `cargo clippy`
4. **Test**: `cargo test`
