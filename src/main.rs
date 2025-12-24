use axum::{
    extract::{State, Form, Json},
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use duckdb::{params, Connection};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, HashSet}, sync::Arc, env};
use chrono::Local;


// The "Truth" derived from the event stream
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LearnedSchema {
    name: String,             // e.g., "InvoiceEvent"
    fields: HashSet<String>,  // e.g., {"amount", "vendor_email"}
    sample_data: String,      // A JSON snapshot for the AI to understand context
}

impl LearnedSchema {
    fn new(name: String, sample_data: String) -> Self {
        Self {
            name,
            fields: HashSet::new(),
            sample_data,
        }
    }
}

// Logic extracted for testing
fn learn_schema(schemas: &mut HashMap<String, LearnedSchema>, type_name: String, payload: &serde_json::Value) {
    let fields: HashSet<String> = payload.as_object()
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    let entry = schemas.entry(type_name.clone()).or_insert_with(|| {
        LearnedSchema::new(
            type_name.clone(),
            serde_json::to_string_pretty(payload).unwrap_or_default(),
        )
    });
    
    entry.fields.extend(fields);
}

fn check_missing_slots(schema: &LearnedSchema, collected_slots: &HashMap<String, String>) -> Vec<String> {
    let mut missing: Vec<String> = schema.fields.iter()
        .filter(|f| !collected_slots.contains_key(*f))
        .cloned()
        .collect();
    missing.sort(); // Deterministic order for tests
    missing
}

// The Chat Session State
#[derive(Clone, Debug, Serialize)]
struct SessionState {
    history: Vec<(String, String)>, // (Role, Content)
    active_intent: Option<String>,  // What schema is the user trying to fill?
    collected_slots: HashMap<String, String>, 
}

// Global Application State
struct AppState {
    // The Database (Embedded OLAP)
    db: Mutex<Connection>,
    // The Dynamic Schema Registry (The Brain)
    schemas: RwLock<HashMap<String, LearnedSchema>>,
    // User Session (Simplified for single user demo)
    session: Mutex<SessionState>,

}

// =================================================================================
// 2. THE FRONTEND (Embedded HTMX)
// =================================================================================

const HTML_TEMPLATE: &str = r##"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The One-File Gateway</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .fade-in { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 h-screen flex overflow-hidden font-mono">

    <div class="w-1/3 border-r border-gray-700 flex flex-col bg-gray-950">
        <div class="p-4 border-b border-gray-800 bg-gray-900">
            <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest">Live Event Stream</h2>
            <div class="text-xs text-green-500 mt-1">Listening on POST /ingest</div>
        </div>
        <div id="event-log" class="flex-1 p-4 overflow-y-auto space-y-2 font-mono text-xs">
            <div class="text-gray-600 italic">Waiting for data...</div>
        </div>
    </div>

    <div class="w-1/3 border-r border-gray-700 flex flex-col relative">
        <div class="p-4 border-b border-gray-800 bg-gray-900">
             <h2 class="text-xs font-bold text-blue-500 uppercase tracking-widest">Agent Interface</h2>
        </div>
        
        <div id="chat-feed" class="flex-1 p-4 overflow-y-auto space-y-4">
            {{ CHAT_HISTORY }}
        </div>

        <div class="p-4 bg-gray-800">
            <form hx-post="/chat" hx-target="#app-root" hx-swap="outerHTML" class="flex gap-2">
                <input type="text" name="msg" autocomplete="off"
                       class="flex-1 bg-gray-900 border border-gray-600 rounded px-3 py-2 focus:border-blue-500 focus:outline-none transition"
                       placeholder="Ask to start a workflow..." autofocus>
                <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded font-bold text-sm">
                    Send
                </button>
            </form>
        </div>
    </div>

    <div class="w-1/3 bg-gray-950 flex flex-col">
        <div class="p-4 border-b border-gray-800 bg-gray-900">
             <h2 class="text-xs font-bold text-purple-500 uppercase tracking-widest">Learned Schemas</h2>
        </div>
        <div class="p-6 space-y-6">
            {{ SCHEMA_VISUALIZATION }}
        </div>
    </div>

</body>
</html>
"##;

// =================================================================================
// 3. THE LOGIC (Main Loop)
// =================================================================================

#[tokio::main]
async fn main() {

    // A. Initialize the "Cortex" (DuckDB)
    let conn = Connection::open_in_memory().expect("Failed to open DuckDB");
    
    // Create a flexible table for raw logs
    conn.execute(
        "CREATE TABLE events (type VARCHAR, payload JSON, received_at TIMESTAMP)",
        params![],
    ).expect("Failed to create table");

    let state = Arc::new(AppState {
        db: Mutex::new(conn),
        schemas: RwLock::new(HashMap::new()),
        session: Mutex::new(SessionState {
            history: vec![("System".into(), "System Online. Waiting for event stream...".into())],
            active_intent: None,
            collected_slots: HashMap::new(),
        }),
    });

    // B. Build the Router
    let app = Router::new()
        .route("/", get(ui_handler))
        .route("/chat", post(chat_handler))
        .route("/ingest", post(ingest_handler))

        .with_state(state);

    println!("⚡ MASTER ONE-FILE ACTIVE");
    println!("   > UI & Chat: http://127.0.0.1:9382");
    println!("   > Ingestion: POST http://127.0.0.1:9382/ingest");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9382").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// =================================================================================
// 4. THE INGESTION ENGINE (The Learning Loop)
// =================================================================================

async fn ingest_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let db = state.db.lock();
    let mut schemas = state.schemas.write();

    // 1. Analyze: What is this?
    let type_name = payload.get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("UnknownEvent")
        .to_string();

    // 2. Persist: Write to DuckDB
    let now = Local::now().to_rfc3339();
    db.execute(
        "INSERT INTO events (type, payload, received_at) VALUES (?, ?, ?)",
        params![type_name, serde_json::to_string(&payload).unwrap(), now],
    ).unwrap();

    // 3. Learn: Update Schema Registry
    // If this is a new event or has new fields, we update our "Brain"
    learn_schema(&mut schemas, type_name.clone(), &payload);

    // 4. Notify UI (HTMX Out-of-Band Swap)
    // This pushes the log to the browser without a refresh!
    let html = format!(
        r#"<div id="event-log" hx-swap-oob="afterbegin">
            <div class="mb-2 p-2 bg-gray-900 rounded border border-gray-700 fade-in">
                <div class="text-green-400 font-bold">{}</div>
                <div class="text-gray-500 truncate">{}</div>
            </div>
           </div>"#, 
        type_name, serde_json::to_string(&payload).unwrap()
    );

    Html(html)
}

// =================================================================================
// 5. THE AGENT (The Chat Logic)
// =================================================================================

#[derive(Deserialize)]
struct ChatInput { msg: String }

#[derive(Debug, Deserialize)]
struct LLMResponse {
    intent: Option<String>,
    slots: Vec<Slot>,
    message: String,
}

#[derive(Debug, Deserialize)]
struct Slot {
    name: String,
    value: String,
}

async fn call_llm(
    input: &str,
    system_prompt: &str
) -> Result<LLMResponse, String> {
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set")?;
    let client = reqwest::Client::new();
    
    // strict: true requires additionalProperties: false on all objects
    let schema = serde_json::json!({
        "name": "gateway_response",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": ["string", "null"],
                    "description": "The name of the schema if an intent is detected, or null."
                },
                "slots": {
                    "type": "array",
                    "description": "Any extracted entity values.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "value": { "type": "string" }
                        },
                        "required": ["name", "value"],
                        "additionalProperties": false
                    }
                },
                "message": {
                    "type": "string",
                    "description": "Response message to the user."
                }
            },
            "required": ["intent", "slots", "message"],
            "additionalProperties": false
        }
    });

    let res = client.post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&serde_json::json!({
            "model": "gpt-4o-2024-08-06", // Ensure version supports Structured Outputs
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": input }
            ],
            "response_format": { 
                "type": "json_schema",
                "json_schema": schema
            }
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !res.status().is_success() {
        return Err(format!("OpenAI Error: {:?}", res.text().await));
    }

    let body: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    let content = body["choices"][0]["message"]["content"].as_str().unwrap_or("{}");
    
    serde_json::from_str(content).map_err(|e| e.to_string())
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Form(input): Form<ChatInput>,
) -> Html<String> {
    // 1. Prepare Context (Hold Locks briefly)
    let system_prompt = {
        let schemas = state.schemas.read();
        let mut session = state.session.lock();
        
        // Update history instantly
        session.history.push(("User".into(), input.msg.clone()));

        let schema_context: Vec<String> = schemas.values().map(|s| {
            format!("- Name: {} (Fields: {:?}, Sample: {})", s.name, s.fields, s.sample_data)
        }).collect();

        format!(
            r#"You are an Agentic Event Gateway. 
            Your goal is to help the user trigger a workflow based on the "Learned Schemas".
            
            KNOWN SCHEMAS:
            {}
    
            CURRENT ACTIVE INTENT: {:?}
    
            INSTRUCTIONS:
            1. Identify if the user wants to start a workflow (Intent Detection).
            2. Extract any entity values provided by the user for the active intent (Slot Filling).
            3. Respond with a JSON object.
            
            OUTPUT FORMAT (JSON ONLY):
            {{
                "intent": "SchemaName" or null,
                "slots": [ {{ "name": "field_name", "value": "value" }} ],
                "message": "Your response to the user"
            }}
            "#,
            schema_context.join("\n"),
            session.active_intent
        )
    };

    // 2. Call LLM (No Locks)
    let llm_result = call_llm(&input.msg, &system_prompt).await;

    // 3. Update State (Re-acquire Locks)
    let response_text = {
         let schemas = state.schemas.read();
         let mut session = state.session.lock();

         match llm_result {
            Ok(llm_res) => {
                // Update Intent
                if let Some(new_intent) = llm_res.intent {
                    if schemas.contains_key(&new_intent) {
                        session.active_intent = Some(new_intent);
                    }
                }

                // Merge Slots
                for slot in llm_res.slots {
                    session.collected_slots.insert(slot.name, slot.value);
                }

                // Gap Analysis
                let mut final_msg = llm_res.message;
                if let Some(intent) = &session.active_intent {
                    if let Some(schema) = schemas.get(intent) {
                        let missing = check_missing_slots(schema, &session.collected_slots);
                        
                        if missing.is_empty() {
                             final_msg.push_str(&format!("<br>✅ <b>Complete!</b> Saved {}.", intent));
                             // Persist to DB if we wanted to...
                             session.active_intent = None;
                             session.collected_slots.clear();
                        } else {
                             final_msg.push_str(&format!("<br><span class='text-red-400'>Missing: {:?}</span>", missing));
                        }
                    }
                }
                final_msg
            },
            Err(e) => format!("⚠️ Error: {}", e)
        }
    };

    // 4. Render
    {
        let mut session = state.session.lock();
        session.history.push(("System".into(), response_text));
    }

    let schemas = state.schemas.read();
    let session = state.session.lock();
    render_full_ui(&session, &schemas)
}

// =================================================================================
// 6. THE RENDERER
// =================================================================================



async fn ui_handler(State(state): State<Arc<AppState>>) -> Html<String> {
    let session = state.session.lock();
    let schemas = state.schemas.read();
    render_full_ui(&session, &schemas)
}

fn render_full_ui(session: &SessionState, schemas: &HashMap<String, LearnedSchema>) -> Html<String> {
    // 1. Render Chat
    let mut chat_html = String::new();
    for (role, content) in &session.history {
        let (color, align) = if role == "User" { ("bg-blue-900", "ml-auto") } else { ("bg-gray-800", "mr-auto") };
        chat_html.push_str(&format!(
            r#"<div class="{} p-3 rounded-lg max-w-xs {} mb-2 text-sm">
                <div class="text-xs opacity-50 mb-1">{}</div>
                <div>{}</div>
            </div>"#, color, align, role, content
        ));
    }

    // 2. Render Brain State (Right Panel)
    let mut brain_html = String::new();
    if schemas.is_empty() {
        brain_html = r#"<div class="text-gray-600 text-center mt-10">Brain Empty.<br>Waiting for data...</div>"#.into();
    } else {
        for schema in schemas.values() {
            // Check if this is the active one
            let is_active = session.active_intent.as_ref() == Some(&schema.name);
            let border = if is_active { "border-blue-500 ring-1 ring-blue-500" } else { "border-gray-800" };
            
            brain_html.push_str(&format!(
                r#"<div class="bg-gray-900 p-4 rounded-lg border {} transition-all">
                    <div class="flex justify-between items-center mb-2">
                        <span class="font-bold text-blue-400">{}</span>
                        <span class="text-xs bg-gray-800 px-2 py-1 rounded">{} fields</span>
                    </div>
                    <div class="text-xs text-gray-500 font-mono break-all">{}</div>
                   </div>"#, 
                border, schema.name, schema.fields.len(), schema.fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
    }

    // 3. Inject
    let output = HTML_TEMPLATE
        .replace("{{ CHAT_HISTORY }}", &chat_html)
        .replace("{{ SCHEMA_VISUALIZATION }}", &brain_html)
        .replace("id=\"app-root\"", ""); // cleanup

    // We wrap the whole thing in a div active for HTMX swapping
    Html(format!("<div id='app-root'>{}</div>", output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_learn_schema_new() {
        let mut schemas = HashMap::new();
        let payload = json!({
            "amount": 100,
            "currency": "USD"
        });
        
        learn_schema(&mut schemas, "Payment".into(), &payload);
        
        let schema = schemas.get("Payment").unwrap();
        assert_eq!(schema.name, "Payment");
        assert!(schema.fields.contains("amount"));
        assert!(schema.fields.contains("currency"));
        assert_eq!(schema.fields.len(), 2);
    }

    #[test]
    fn test_learn_schema_merge() {
        let mut schemas = HashMap::new();
        
        // First event
        let payload1 = json!({"field1": "a"});
        learn_schema(&mut schemas, "TestEvent".into(), &payload1);
        
        // Second event with new field
        let payload2 = json!({"field1": "b", "field2": "c"});
        learn_schema(&mut schemas, "TestEvent".into(), &payload2);
        
        let schema = schemas.get("TestEvent").unwrap();
        assert!(schema.fields.contains("field1"));
        assert!(schema.fields.contains("field2"));
        assert_eq!(schema.fields.len(), 2);
    }

    #[test]
    fn test_check_missing_slots() {
        let mut schema = LearnedSchema::new("Test".into(), "{}".into());
        schema.fields.insert("email".into());
        schema.fields.insert("name".into());
        
        let mut collected = HashMap::new();
        
        // Case 1: All missing
        let missing = check_missing_slots(&schema, &collected);
        assert_eq!(missing, vec!["email", "name"]);

        // Case 2: Partial
        collected.insert("email".into(), "bob@test.com".into());
        let missing = check_missing_slots(&schema, &collected);
        assert_eq!(missing, vec!["name"]);

        // Case 3: Complete
        collected.insert("name".into(), "Bob".into());
        let missing = check_missing_slots(&schema, &collected);
        assert!(missing.is_empty());
    }
}
