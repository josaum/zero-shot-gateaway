# Build, Lint, and Test Commands
- **Build**: `cargo build --features oar`
- **Run**: `cargo run --bin one_file_gateway --features oar`
- **Check**: `cargo check`
- **Lint**: `cargo clippy`
- **Test (All)**: `cargo test`
- **Test (Single)**: `cargo test -- <test_name_substring>`

# Code Style & Conventions
- **Architecture**: STRICTLY maintain the "One-File" architecture in `src/main.rs`. Do not split into multiple files/modules unless explicitly instructed.
- **Formatting**: Run `cargo fmt` on changes. Use standard Rust indentation (4 spaces).
- **Naming**: `snake_case` for variables/functions, `PascalCase` for Structs/Enums.
- **Concurrency**: Use `Arc<AppState>` pattern with `parking_lot::Mutex` or `RwLock` for shared state.
- **Imports**: Group imports by crate (std, external). Keep imports clean at the top of `src/main.rs`.
- **Error Handling**: Use `Result` for fallible operations in handlers. `unwrap()`/`expect()` are acceptable for startup initialization or strict prototypes but prefer `match` or `?` in production code.
- **Dependencies**: New dependencies must be added to `Cargo.toml`.
- **LLM Integration**: Maintain strict JSON schema definitions for OpenAI calls.
- **Reflexivity**: The system reacts to `SystemConfig` events to update its own configuration (webhook_url, ontology_iri).
- **Persistence**: Full ontology (TTL) and ABox (JSON-LD) are emitted to the configured webhook on every state change.
