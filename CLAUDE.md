# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build --release    # Recommended for TUI performance
cargo run --release      # Run the TUI Gateway
cargo check              # Type checking only
cargo clippy             # Lint with warnings
cargo fmt                # Format code
curl -X POST ...         # Use curl to trigger ingestion/chat API
```

## Critical Constraint: TUI & Arrow Architecture

1.  **TUI-First**: The primary interface is the Terminal (`ratatui`). Visual updates must be efficient.
2.  **Arrow IPC**: We use **Zero-Copy** rendering. `tui.rs` consumes `arrow::record_batch::RecordBatch` directly from DuckDB. Do not introduce intermediate structs for event data display if possible.
3.  **Real-time**: Use `tokio::sync::broadcast` for updates. Do not introduce polling loops.

## Architecture

Single-binary gateway combining:
- **Axum**: Async web server (background, port 9382).
- **DuckDB**: Embedded OLAP with `vtab-arrow` feature.
- **Ratatui**: TUI frontend with Vim-like navigation.
- **Gemini API**: AI intent detection and chat.
- **Tokio Broadcast**: Internal signaling bus.

## Concurrency Pattern

```rust
struct AppState {
    db: Mutex<Connection>,           // DuckDB (Arrow enabled)
    schemas: RwLock<HashMap<...>>,   // Schema registry
    session: Mutex<SessionState>,    // Chat session
    tx_notify: broadcast::Sender<()>, // Real-time signal
    // ...
}
```

## Key Flows

**Ingestion**: 
`POST /ingest` → Parse → Persist DuckDB → `tx_notify.send()` → Ack

**TUI Update**: 
`tokio::select!` → `rx_notify.recv()` → `db.query_arrow()` → `term.draw()`

**AI Chat**: 
TUI Input → `POST /api/chat` → Gemini API → Intent/Slots → Response → Broadcast

## Environment

Requires `GEMINI_API_KEY` in `.env`.

## Code Style

- `snake_case` for variables/functions.
- `tui.rs`: Handles all rendering and usage input.
- `main.rs`: Handles API routes, database wiring, and state.
