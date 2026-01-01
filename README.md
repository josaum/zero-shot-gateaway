# ⚡ Master One-File Event Gateway

[![CI](https://github.com/josaum/zero-shot-gateaway/actions/workflows/ci.yml/badge.svg)](https://github.com/josaum/zero-shot-gateaway/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub repo size](https://img.shields.io/github/repo-size/josaum/zero-shot-gateaway)](https://github.com/josaum/zero-shot-gateaway)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)
[![Code Style](https://img.shields.io/badge/code__style-rustfmt-blue)](https://github.com/rust-lang/rustfmt)


> **The "God Node" of Event Processing**: A single Rust binary that acts as an Ingestion Engine, OLAP Database, AI Agent, and Formal Ontology Reasoner.

This project demonstrates a radical **"One-File"** architecture where a highly capable agentic system is contained within a single `main.rs`, integrating:
*   **Rust (Axum)**: High-performance async web server.
*   **DuckDB**: Embedded analytical database (OLAP).
*   **OpenAI (GPT-4o)**: Strict JSON intent detection and slot filling.
*   **HTMX / Tailwind**: Server-side rendered, zero-build frontend.

---

## 🚀 Features

1.  **Learned Schemas**: Send any JSON event to `/ingest`. The system automatically learns the schema, tracks fields, and keeps a sample.
2.  **Reflexive & Persistence**: Configuring the system via `SystemConfig` events triggers immediate self-reconfiguration. It emits the full learned ontology (TTL) and data history (JSON-LD) to a webhook on every change.
3.  **Resilient Exports**: Exports are batched (default: 5 events) with automatic retry and exponential backoff for reliability.
4.  **Observability**: Structured logging with `/metrics` and `/health` endpoints for production monitoring.
5.  **Context-Aware Chat**: Talk to the system in natural language. It uses the learned schemas to understand what you want (Intent Detection) and extracts the necessary details (Slot Filling).
6.  **Zero-Latency UI**: A real-time dashboard powered by HTMX that updates instantly as events arrive or chat occurs.

---

## 🛠 Prerequisites

### 1. Rust
Ensure you have the latest stable Rust installed.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. OpenAI API Key
Required for the "Agent" intelligence.
```bash
export OPENAI_API_KEY="sk-..."
```

---

## 🏃‍♂️ Running the Gateway

Because this uses a single binary, you just need to run:

```bash
# Run!
cargo run --bin one_file_gateway
```

Once running, open your browser to: **`http://127.0.0.1:9382`**

### Endpoints
- **UI & Chat**: `http://127.0.0.1:9382`
- **Ingestion**: `POST http://127.0.0.1:9382/ingest`
- **Config**: `POST http://127.0.0.1:9382/config`
- **Health**: `GET http://127.0.0.1:9382/health`
- **Metrics**: `GET http://127.0.0.1:9382/metrics`

---

## 📂 Project Structure

```
/
├── src/main.rs           # 🧠 THE BRAIN: All logic, UI, and DB code.
├── Cargo.toml            # Dependencies
├── AGENTS.md             # Developer guide & commands
├── README.md             # This file
├── llms.txt              # AI Context file
├── LICENSE               # MIT License
└── .github/workflows     # CI Pipeline
```

> [!IMPORTANT]
> **One-File Rule**: All application logic MUST remain in `src/main.rs`. Do not refactor into modules. This is a purposeful architectural constraint.

---

## 📖 Usage Examples

### 1. Ingestion (The Learning Phase)
Feed the system some data. It will "learn" that these event types exist.

**Configure via UI:**
Use the Config panel in the sidebar to set webhook URL and batch size.

**Configure via API (Reflexivity):**
First, tell the system where to send the ontology.
```bash
curl -X POST http://127.0.0.1:9382/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "type": "SystemConfig",
    "webhook_url": "https://webhook.site/your-id",
    "ontology_iri": "http://my-org.com/ontology/"
  }'
```
*The system immediately reacts by exporting its current state to the webhook.*

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

**Ingest a `PaymentProcessed` event:**
```bash
curl -X POST http://127.0.0.1:9382/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "type": "PaymentProcessed",
    "amount": 99.99,
    "currency": "USD",
    "transaction_id": "tx_12345"
  }'
```

*Check the UI: You will see these schemas appear in the "Learned Schemas" panel.*

---

### 2. The Agent (The Acting Phase)
Go to the chat interface in the browser to interact with the agent. It knows about the events you just sent.

**User:** "I want to manually record a new signup."
**Agent:** "I can help with that. I detected you want to trigger a `UserSignup`. Please provide the `email` and `plan`."

**User:** "It's for bob@test.com and he is on the Starter plan."
**Agent:** "Captured `email`=bob@test.com, `plan`=Starter. I still need the `region`."

---

## 📊 Monitoring

### Health Check
```bash
curl http://127.0.0.1:9382/health
# Returns: OK
```

### Metrics
```bash
curl http://127.0.0.1:9382/metrics
# Returns JSON:
# {
#   "events_ingested": 42,
#   "schemas_learned": 3,
#   "exports_attempted": 8,
#   "exports_succeeded": 7,
#   "exports_failed": 1,
#   "last_export_at": "2025-12-24T21:30:00+00:00",
#   "pending_exports": 2,
#   "webhook_configured": true,
#   "export_batch_size": 5
# }
```

---

## 🧠 Architecture Notes

- **One File**: Everything (DB init, UI HTML, API routes, Logic) is in `src/main.rs`.
- **Concurrency**:
    - `AppState` is wrapped in `Arc` and shared across threads.
    - `Mutex` protects the DuckDB connection and Session state.
    - `RwLock` protects the Schema Registry for high-concurrency reads.
- **Export Strategy**: Events are queued and exported in batches. Failed exports are retried 3 times with exponential backoff.
- **Strict JSON**: We use OpenAI's `response_format: { type: "json_schema", ... }` to ensure the LLM *always* responds with valid machine-readable JSON, eliminating parsing errors.

---

## ❓ Troubleshooting

### Port 9382 in use
If you see `Os { code: 48, kind: AddrInUse, message: "Address already in use" }`:
1. Check if another instance is running.
2. Kill the process occupying port 9382:
   ```bash
   lsof -i :9382
   kill -9 <PID>
   ```

### Linker Errors
If you experience linker errors on macOS, ensure you have the command line tools installed:
```bash
xcode-select --install
```

---

## 👥 Contributing

1. **Keep it One-File**: Do not split `src/main.rs`.
2. **Format Code**: Run `cargo fmt` before committing.
3. **Check Clippy**: Run `cargo clippy`.
4. **Test**: Run `cargo test`.
