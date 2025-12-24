# ⚡ Master One-File Event Gateway

> **The "God Node" of Event Processing**: A single Rust binary that acts as an Ingestion Engine, OLAP Database, AI Agent, and Formal Ontology Reasoner.

This project demonstrates a radical "One-File" architecture where a highly capable agentic system is contained within a single `main.rs`, integrating:
*   **Rust (Axum)**: High-performance async web server.
*   **DuckDB**: Embedded analytical database (OLAP).
*   **OpenAI (GPT-4o)**: Strict JSON intent detection and slot filling.
*   **HTMX / Tailwind**: Server-side rendered, zero-build frontend.


---

## 🚀 Features

1.  **Learned Schemas**: Send any JSON event to `/ingest`. The system automatically learns the schema, tracks fields, and keeps a sample.
2.  **Context-Aware Chat**: Talk to the system in natural language. It uses the learned schemas to understand what you want (Intent Detection) and extracts the necessary details (Slot Filling).

4.  **Zero-Latency UI**: A real-time dashboard powered by HTMX that updates instantly as events arrive or chat occurs.

---

## 🛠 Prerequisites

### 1. Rust
Ensure you have the latest stable Rust installed.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 3. OpenAI API Key
Required for the implementation of the "Agent" intelligence.
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

Once running, open your browser to: **`http://127.0.0.1:3000`**

---

## 📖 Usage Examples

### 1. Ingestion (The Learning Phase)
Feed the system some data. It will "learn" that these event types exist.

**Ingest a `UserSignup` event:**
```bash
curl -X POST http://127.0.0.1:3000/ingest \
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
curl -X POST http://127.0.0.1:3000/ingest \
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
Go to the chat interface in the browser (or use curl) to interact with the agent. It knows about the events you just sent.

**User:** "I want to manually record a new signup."
**Agent:** "I can help with that. I detected you want to trigger a `UserSignup`. Please provide the `email` and `plan`."

**User:** "It's for bob@test.com and he is on the Starter plan."
**Agent:** "Captured `email`=bob@test.com, `plan`=Starter. I still need the `region`."


---

## 🧠 Architecture Notes

- **One File**: Everything (DB init, UI HTML, API routes, Logic) is in `src/main.rs`.
- **Concurrency**: 
    - `AppState` is wrapped in `Arc` and shared across threads.
    - `Mutex` protects the DuckDB connection and Session state.
    - `RwLock` protects the Schema Registry for high-concurrency reads.

- **Strict JSON**: We use OpenAI's `response_format: { type: "json_schema", ... }` to ensure the LLM *always* responds with valid machine-readable JSON, eliminating parsing errors.

