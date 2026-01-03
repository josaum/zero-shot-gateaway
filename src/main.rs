use clap::Parser;
use htmlescape::encode_minimal;
use axum::{
    extract::{Request, State, Json, Path},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
    http::{StatusCode, header::AUTHORIZATION},
};
use duckdb::{params, Connection};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, HashSet}, sync::Arc, env, time::Duration};
use chrono::Local;
use tokio::sync::broadcast;
use tracing::{info, warn, error};

mod tui;
mod collider;
mod models;
mod schema_registry;
#[cfg(feature = "oar")]
mod ocr_pipeline;
mod gliner;

use collider::{LockedCollider, ConversationEvent, ColliderMetrics};
use gliner::GlinerModel;


// The "Truth" derived from the event stream
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LearnedSchema {
    name: String,             // e.g., "InvoiceEvent"
    fields: HashSet<String>,  // e.g., {"amount", "vendor_email"}
    sample_data: String,      // A JSON snapshot for the AI to understand context
    #[serde(default)]
    jsonld_context: Option<serde_json::Value>, // Captured @context
}

impl LearnedSchema {
    fn new(name: String, sample_data: String, jsonld_context: Option<serde_json::Value>) -> Self {
        Self {
            name,
            fields: HashSet::new(),
            sample_data,
            jsonld_context,
        }
    }
}

// Logic extracted for testing
fn learn_schema(schemas: &mut HashMap<String, LearnedSchema>, type_name: String, payload: &serde_json::Value) {
    let fields: HashSet<String> = payload.as_object()
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();
    
    // Capture @context if present
    let context = payload.get("@context").cloned();

    let entry = schemas.entry(type_name.clone()).or_insert_with(|| {
        LearnedSchema::new(
            type_name.clone(),
            serde_json::to_string_pretty(payload).unwrap_or_default(),
            context
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

// Dynamic System Configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Config {
    webhook_url: Option<String>,
    ontology_iri: String,
    #[serde(default)]
    export_batch_size: usize, 
    #[serde(default)]
    export_interval_secs: u64,
}

// Global Application State
#[derive(Debug)]
struct AppState {
    // The Database (Embedded OLAP)
    db: Mutex<Connection>,
    // The Dynamic Schema Registry (The Brain)
    schemas: RwLock<HashMap<String, LearnedSchema>>,
    // User Session (Simplified for single user demo)
    session: Mutex<SessionState>,
    // System Configuration
    config: RwLock<Config>,
    // Export Queue for retry logic
    export_queue: Mutex<Vec<(String, serde_json::Value)>>,
    // Metrics
    metrics: Mutex<Metrics>,
    // Auth Token
    auth_token: Option<String>,
    // Real-time notification channel
    tx_notify: broadcast::Sender<()>,
    // Physics Engine Collider
    collider: Arc<LockedCollider>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
   /// Optional path to a persistent DuckDB file (e.g., ./gateway.db)
   #[arg(long)]
   db_path: Option<String>,

   /// Optional Bearer token for securing /ingest and /config
   #[arg(long)]
   token: Option<String>,

   /// Run without the Terminal UI (headless mode)
   #[arg(long)]
   no_tui: bool,

   /// Path or URL for the Embedding Model (ONNX)
   #[arg(long, env = "EMBEDDING_MODEL")]
   embedding_model: Option<String>,

   /// Path or URL for the Tokenizer (JSON)
   #[arg(long, env = "TOKENIZER_MODEL")]
   tokenizer_model: Option<String>,

   /// Path or URL for OCR Table Cell Detection Model (RT-DETR)
   #[arg(long, env = "OCR_TABLE_CELL_MODEL")]
   ocr_table_cell_model: Option<String>,

   /// Path or URL for PaddleOCR Detection Model (ONNX)
   #[arg(long, env = "OCR_DET_MODEL")]
   ocr_det_model: Option<String>,

   /// Path or URL for PaddleOCR Recognition Model (ONNX)
   #[arg(long, env = "OCR_REC_MODEL")]
   ocr_rec_model: Option<String>,

   /// Path or URL for PaddleOCR Layout Model (ONNX) - Optional/Manual
   #[arg(long, env = "OCR_LAYOUT_MODEL")]
   ocr_layout_model: Option<String>,

   /// Path or URL for PaddleOCR Table Model (SLANet ONNX) - Optional/Manual
   #[arg(long, env = "OCR_TABLE_MODEL")]
   ocr_table_model: Option<String>,

   /// Path or URL for OCR Character Dictionary (txt)
   #[arg(long, env = "OCR_KEYS_MODEL")]
   ocr_keys_model: Option<String>,

   /// Path or URL for GLiNER Model (NER / Structured Extraction)
   #[arg(long, env = "OCR_NER_MODEL", default_value = "models/gliner_multi_v2.1")]
   ocr_ner_model: String,

   /// OCR Pipeline to use: legacy, oar-structure, oar-vl
   #[arg(long, env = "OCR_PIPELINE", default_value = "oar-structure")]
   ocr_pipeline: String,

   /// Path to PaddleOCR-VL model directory (for oar-vl pipeline)
   #[arg(long, env = "VL_MODEL_DIR")]
   vl_model_dir: Option<String>,
    /// Path to GLiNER model directory (containing .onnx and tokenizer.json)
    #[arg(long, default_value = "models/gliner_multi_v2.1")]
    gliner_dir: String,
}

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // If no token is configured, allow everything
    if state.auth_token.is_none() {
        return Ok(next.run(req).await);
    }

    let token = state.auth_token.as_ref().unwrap();
    let auth_header = req.headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    match auth_header {
        Some(t) if t == token => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

#[derive(Clone, Debug, Default)]
struct Metrics {
    events_ingested: usize,
    schemas_learned: usize,
    exports_attempted: usize,
    exports_succeeded: usize,
    exports_failed: usize,
    last_export_at: Option<String>,
    _collider_metrics: Option<ColliderMetrics>,
}


// =================================================================================
// 3. THE LOGIC (Main Loop)
// =================================================================================

#[tokio::main]
async fn main() {
    // Load .env file if it exists (for GEMINI_API_KEY, etc.)
    let _ = dotenvy::dotenv();
    
    let args = Args::parse();
    let no_tui = args.no_tui;

    // Setup tracing
    tracing_subscriber::fmt().init();
    info!("⚡ MASTER ONE-FILE ACTIVE");

    // A. Initialize the "Cortex" (DuckDB)
    let conn = if let Some(path) = &args.db_path {
        info!("📂 Opening persistent database at: {}", path);
        Connection::open(path).expect("Failed to open DuckDB")
    } else {
        info!("⚠️  Using IN-MEMORY database (Data will be lost on exit)");
        Connection::open_in_memory().expect("Failed to open DuckDB")
    };
    
    // Create a flexible table for raw logs
    // We use IF NOT EXISTS now because the DB might persist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events (type VARCHAR, payload JSON, received_at TIMESTAMP)",
        params![],
    ).expect("Failed to create table");

    // B. Model Lifecycle (Bootstrap)
    let embedding_model_target = args.embedding_model.unwrap_or_else(|| "https://huggingface.co/Xenova/bge-m3/resolve/main/onnx/model_quantized.onnx".to_string());
    let tokenizer_target = args.tokenizer_model.unwrap_or_else(|| "https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json".to_string());

    let model_path = if embedding_model_target.starts_with("http") {
         models::ModelManager::ensure_model("bge-m3-int8.onnx", &embedding_model_target, std::path::Path::new("models/bge-m3-int8.onnx")).expect("Failed to download embedding model")
    } else {
         std::path::PathBuf::from(embedding_model_target)
    };

    let tokenizer_path = if tokenizer_target.starts_with("http") {
         models::ModelManager::ensure_model("tokenizer.json", &tokenizer_target, std::path::Path::new("models/tokenizer.json")).expect("Failed to download tokenizer")
    } else {
         std::path::PathBuf::from(tokenizer_target)
    };

    let ocr_det_target = args.ocr_det_model.unwrap_or_else(|| "https://huggingface.co/marsena/paddleocr-onnx-models/resolve/main/PP-OCRv5_server_det_infer.onnx".to_string());
    let ocr_rec_target = args.ocr_rec_model.unwrap_or_else(|| "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/rec.onnx".to_string());

    let ocr_det_path = if ocr_det_target.starts_with("http") {
        models::ModelManager::ensure_model("ocr_det.onnx", &ocr_det_target, std::path::Path::new("models/ocr_det.onnx")).expect("Failed to download OCR Detection model")
    } else {
        std::path::PathBuf::from(ocr_det_target)
    };

    let ocr_rec_path = if ocr_rec_target.starts_with("http") {
        models::ModelManager::ensure_model("ocr_rec.onnx", &ocr_rec_target, std::path::Path::new("models/ocr_rec.onnx")).expect("Failed to download OCR Recognition model")
    } else {
        std::path::PathBuf::from(ocr_rec_target)
    };

    let ocr_layout_target = args.ocr_layout_model.or_else(|| {
         if std::path::Path::new("models/picodet_layout_1x.onnx").exists() {
             Some("models/picodet_layout_1x.onnx".to_string())
         } else if std::path::Path::new("models/layout.onnx").exists() {
             Some("models/layout.onnx".to_string())
         } else {
             Some("https://huggingface.co/cycloneboy/picodet_lcnet_x1_0_fgd_layout_infer/resolve/main/model.onnx".to_string())
         }
    }).unwrap();
    
    let ocr_layout_path = if ocr_layout_target.starts_with("http") {
         models::ModelManager::ensure_model("layout.onnx", &ocr_layout_target, std::path::Path::new("models/layout.onnx")).ok()
    } else {
         Some(std::path::PathBuf::from(ocr_layout_target))
    };

    let ocr_table_target = args.ocr_table_model.or_else(|| {
        if std::path::Path::new("models/table_structure.onnx").exists() {
            Some("models/table_structure.onnx".to_string())
        } else if std::path::Path::new("models/SLANet.onnx").exists() {
            Some("models/SLANet.onnx".to_string())
        } else {
            Some("https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/TabRec/SlanetPlus/slanet-plus.onnx".to_string())
        }
    });

    let ocr_table_path = if let Some(target) = ocr_table_target {
        if target.starts_with("http") {
             models::ModelManager::ensure_model("table_structure.onnx", &target, std::path::Path::new("models/table_structure.onnx")).ok()
        } else {
             Some(std::path::PathBuf::from(target))
        }
    } else {
        None
    };

    let ocr_table_cell_path = if let Some(target) = args.ocr_table_cell_model {
        if target.starts_with("http") {
             models::ModelManager::ensure_model("table_cell_det.onnx", &target, std::path::Path::new("models/table_cell_det.onnx"))
                .ok()
                .map(|p| p.to_string_lossy().to_string())
        } else {
             Some(target)
        }
    } else {
        if std::path::Path::new("models/table_cell_det.onnx").exists() {
             Some("models/table_cell_det.onnx".to_string())
        } else {
             None
        }
    };

    let ocr_keys_target = args.ocr_keys_model.unwrap_or_else(|| "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.6/ppocr/utils/dict/latin_dict.txt".to_string());
    
    let ocr_keys_path = if ocr_keys_target.starts_with("http") {
         models::ModelManager::ensure_model("ocr_keys.txt", &ocr_keys_target, std::path::Path::new("models/ocr_keys.txt")).expect("Failed to download OCR keys")
    } else {
         std::path::PathBuf::from(ocr_keys_target)
    };

    let (tx_notify, _) = broadcast::channel(100);

    // C. Initialize OCR Pipeline
    #[cfg(feature = "oar")]
    let pipeline_type: ocr_pipeline::PipelineType = args.ocr_pipeline.parse()
        .unwrap_or_else(|e| {
            warn!("⚠️  Invalid OCR pipeline, falling back to legacy: {}", e);
            ocr_pipeline::PipelineType::Legacy
        });

    #[cfg(feature = "oar")]
    let ocr_pipeline: Option<Box<dyn ocr_pipeline::OcrPipelineTrait>> = match pipeline_type {
        ocr_pipeline::PipelineType::Legacy => {
            info!("📄 Using Legacy OCR Pipeline (custom PP-StructureV3)");
            Some(Box::new(ocr_pipeline::LegacyPipeline))
        }
        ocr_pipeline::PipelineType::OarStructure => {
            info!("📄 Using OAR Structure Pipeline");
            let layout_model = ocr_layout_path.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| "models/picodet_layout_1x.onnx".into());
            info!("🛠️  Layout Model Path: '{}'", layout_model);
            let table_structure = ocr_table_path.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| "models/slanext_wired.onnx".into());
            let table_dict = "models/table_structure_dict_ch.txt"; 

            match ocr_pipeline::oar_structure::OarStructurePipeline::new(
                &layout_model,
                Some("models/pp-lcnet_x1_0_table_cls.onnx"),
                ocr_table_cell_path.as_deref(),
                &table_structure,
                table_dict,
                &ocr_det_path.to_string_lossy(),
                &ocr_rec_path.to_string_lossy(),
                &ocr_keys_path.to_string_lossy(),
            ) {
                Ok(p) => Some(Box::new(p)),
                Err(e) => {
                    warn!("⚠️  Failed to initialize OAR Structure Pipeline: {}. Falling back to legacy.", e);
                    Some(Box::new(ocr_pipeline::LegacyPipeline))
                }
            }
        }
        #[cfg(feature = "vl")]
        ocr_pipeline::PipelineType::OarVL => {
            info!("📄 Using OAR VL Pipeline (PaddleOCR-VL)");
            let model_dir = args.vl_model_dir.unwrap_or_else(|| "models/PaddleOCR-VL".into());
            match ocr_pipeline::oar_vl::OarVLPipeline::new(&model_dir, true) {
                Ok(p) => Some(Box::new(p)),
                Err(e) => {
                    warn!("⚠️  Failed to initialize OAR VL Pipeline: {}. Falling back to legacy.", e);
                    Some(Box::new(ocr_pipeline::LegacyPipeline))
                }
            }
        }
        #[allow(unreachable_patterns)]
        _ => {
            warn!("⚠️  Selected pipeline feature not enabled, fallback to legacy");
            Some(Box::new(ocr_pipeline::LegacyPipeline))
        }
    };
    
    #[cfg(not(feature = "oar"))]
    let ocr_pipeline: Option<Box<dyn std::any::Any>> = None; // Dummy type

    // 7. Initialize GLiNER
    #[cfg(feature = "oar")]
    let gliner_model = if std::path::Path::new(&args.gliner_dir).exists() {
        info!("🧠 Loading GLiNER Model from: {}", args.gliner_dir);
        match crate::gliner::GlinerModel::new(&args.gliner_dir) {
            Ok(m) => Some(m),
            Err(e) => {
                warn!("⚠️  Failed to load GLiNER: {}. NER will be disabled.", e);
                None
            }
        }
    } else {
        warn!("⚠️  GLiNER directory not found at {}. NER disabled.", args.gliner_dir);
        None
    };

    #[cfg(not(feature = "oar"))]
    let gliner_model: Option<()> = None;

    let state = Arc::new(AppState {
        db: Mutex::new(conn),
        schemas: RwLock::new(HashMap::new()),
        session: Mutex::new(SessionState {
            history: vec![("System".into(), "Gateway initialized. Listening for events...".into())],
            active_intent: None,
            collected_slots: HashMap::new(),
        }),
        config: RwLock::new(Config {
            webhook_url: None,
            ontology_iri: "http://example.org/ontology/".to_string(),
            export_batch_size: 5,
            export_interval_secs: 30,
        }),
        export_queue: Mutex::new(Vec::new()),
        metrics: Mutex::new(Metrics::default()),
        auth_token: args.token,
        tx_notify: tx_notify.clone(),
        collider: {
            #[cfg(feature = "oar")]
            {
               LockedCollider::new(
                   ocr_pipeline, 
                   gliner_model,
                   model_path.to_str(),
                   tokenizer_path.to_str()
               ).unwrap_or_else(|e| {
                   panic!("Physics Engine critical failure: {}", e);
               }) 
            }
            #[cfg(not(feature = "oar"))]
            {
               LockedCollider::new(
                   None,
                   model_path.to_str(),
                   tokenizer_path.to_str()
               ).unwrap_or_else(|e| {
                   panic!("Physics Engine critical failure: {}", e);
               })
            }
        },
    });

    // Graceful shutdown signal handler
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("Received shutdown signal");
    });

    // B. Build the Router
    // 1. Protected Routes (Ingest & Config)
    let protected = Router::new()
        .route("/api/events", get(list_events_handler))
        .route("/api/schemas", get(list_schemas_handler))
        .route("/api/schemas/:name", get(get_schema_handler))
        .route("/ingest", post(ingest_handler))
        .route("/api/config", post(config_handler))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // 2. Public Routes & Merge
    let app = Router::new()
        .route("/api/chat", post(chat_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .merge(protected)
        .with_state(state.clone());

    // Spawn API server in background
    tokio::spawn(async move {
        info!("   > API Server: http://127.0.0.1:9382");
        info!("   > Ingestion: POST http://127.0.0.1:9382/ingest");
        info!("   > Chat: POST http://127.0.0.1:9382/api/chat");
        info!("   > Health: GET http://127.0.0.1:9382/health");
        info!("   > Metrics: GET http://127.0.0.1:9382/metrics");
        
        let listener = tokio::net::TcpListener::bind("0.0.0.0:9382").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    // Run TUI in foreground (or wait in headless mode)
    if no_tui {
        info!("🚀 Running in headless mode (use --help for TUI mode)");
        // Wait forever
        std::future::pending::<()>().await;
    } else {
        info!("🖥️  Launching Terminal UI...");
        if let Err(e) = tui::run_tui(state).await {
            error!("TUI error: {}", e);
        }
    }
}

// =================================================================================
// 4. THE INGESTION ENGINE (The Learning Loop)
// =================================================================================

#[derive(Deserialize)]
struct IngestPayload {
    #[serde(rename = "type")]
    event_type: Option<String>,
    webhook_url: Option<String>,
    ontology_iri: Option<String>,
    #[serde(flatten)]
    payload: HashMap<String, serde_json::Value>,
}

async fn perform_export(state: Arc<AppState>) {
    let (url, ontology_iri, events, schemas_data) = {
        let config = state.config.read();
        let mut queue = state.export_queue.lock();
        
        if queue.is_empty() {
            return;
        }

        let events = queue.drain(..).collect::<Vec<_>>();
        let url = config.webhook_url.clone();
        let ontology_iri = config.ontology_iri.clone();
        
        let schemas = state.schemas.read();
        let schemas_data: Vec<(String, HashSet<String>, Option<serde_json::Value>, String)> = schemas.iter()
            .map(|(name, s)| (name.clone(), s.fields.clone(), s.jsonld_context.clone(), s.sample_data.clone()))
            .collect();
        
        (url, ontology_iri, events, schemas_data)
    };

    if let Some(url) = url {
        info!("📤 Exporting {} events to {}", events.len(), url);
        
        // Update metrics
        {
            let mut m = state.metrics.lock();
            m.exports_attempted += events.len();
        }
        
        // 1. Generate TBox (Turtle) with proper RDF
        let mut ttl = format!("@prefix : <{}> .\n", ontology_iri);
        ttl.push_str("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n");
        ttl.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        ttl.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");
        
        for (name, fields, _, _) in &schemas_data {
            ttl.push_str(&format!(":{} a owl:Class ;\n", name));
            ttl.push_str(&format!("    rdfs:label \"{}\" ;\n", name));
            for field in fields {
                ttl.push_str(&format!("    :hasField \"{}\" ;\n", field));
                ttl.push_str(&format!("    rdfs:domain :{} .\n\n", name));
            }
        }

        // 2. Generate ABox (JSON-LD) with proper @context
        let abox: Vec<serde_json::Value> = schemas_data.iter().flat_map(|(_, _, _, sample)| {
            serde_json::from_str::<serde_json::Value>(sample).ok()
        }).collect();

        let abox = serde_json::json!({
            "@context": ontology_iri,
            "@type": "@graph",
            "@graph": abox
        });

        // 3. Payload
        let payload = serde_json::json!({
            "ontology_ttl": ttl,
            "abox_jsonld": abox,
            "exported_at": Local::now().to_rfc3339(),
            "event_count": events.len()
        });

        // 4. Send with retry logic
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to build reqwest client");
        
        for attempt in 1..=3 {
            match client.post(&url).json(&payload).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("✅ Export success");
                    let mut m = state.metrics.lock();
                    m.exports_succeeded += events.len();
                    m.last_export_at = Some(Local::now().to_rfc3339());
                    break;
                }
                Ok(resp) => {
                    warn!("Export attempt {} failed: HTTP {}", attempt, resp.status());
                }
                Err(e) => {
                    warn!("Export attempt {} failed: {}", attempt, e);
                }
            }
            if attempt < 3 {
                tokio::time::sleep(Duration::from_secs(2_u64.pow(attempt as u32))).await;
            }
        }
    }
}



async fn ingest_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    // 1. Parsing
    let parsed: IngestPayload = match serde_json::from_value(payload.clone()) {
        Ok(p) => p,
        Err(_) => {
            error!("Failed to parse ingest payload");
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Invalid payload"}))).into_response();
        }
    };
    
    // 2. AI Enrichment
    let mut type_name = parsed.event_type.clone().unwrap_or_else(|| "UnknownEvent".into());
    
    if type_name == "UnknownEvent" {
        let payload_str_for_llm = serde_json::to_string(&parsed.payload).unwrap_or_default();
        let system_prompt = "You are an intelligent data intake agent. Analyze the provided JSON webhook payload and infer a concise, PascalCase EventType name (e.g., 'UserSignup', 'GitHubPush', 'WhatsAppMessage'). Return it in the 'intent' field. If unsure, use 'GenericWebhook'.";
        
        match call_llm(&payload_str_for_llm, system_prompt).await {
            Ok(ur) => {
                if let Some(inferred) = ur.intent {
                    info!("🤖 Gemini inferred event type: {} -> {}", type_name, inferred);
                    type_name = inferred;
                }
            },
            Err(e) => {
                warn!("⚠️ LLM inference failed: {}", e);
            }
        }
    }

    // 2.5 Metrics
    {
        let mut m = state.metrics.lock();
        m.events_ingested += 1;
    }

    // 2.6 Dynamic Config
    if type_name == "SystemConfig" {
        let mut config = state.config.write();
        let mut updated = false;
        
        if let Some(url) = parsed.webhook_url {
            config.webhook_url = Some(url.clone());
            info!("🔗 Webhook updated: {}", url);
            updated = true;
        }
        if let Some(iri) = parsed.ontology_iri {
            config.ontology_iri = iri.clone();
            info!("🧠 Ontology IRI updated: {}", iri);
            updated = true;
        }
        
        if updated {
            return (StatusCode::OK, Json(serde_json::json!({"status": "configuration_updated"}))).into_response();
        }
    }

    // 3. Persist
    let now = Local::now().to_rfc3339();
    let payload_str = serde_json::to_string(&payload).unwrap_or_default();
    
    {
        let db = state.db.lock();
        if let Err(e) = db.execute(
            "INSERT INTO events (type, payload, received_at) VALUES (?, ?, ?)",
            params![&type_name, &payload_str, &now],
        ) {
            error!("Failed to persist event: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Database error"}))).into_response();
        }
    }

    // 4. Learn
    let schema_was_new; 
    {
        let mut schemas = state.schemas.write();
        schema_was_new = !schemas.contains_key(&type_name);
        learn_schema(&mut schemas, type_name.clone(), &payload);
    }
    
    if schema_was_new {
        let mut m = state.metrics.lock();
        m.schemas_learned += 1;
    }

    // 5. Export
    {
        let mut queue = state.export_queue.lock();
        queue.push((type_name.clone(), payload));
        
        let config = state.config.read();
        let should_export = queue.len() >= config.export_batch_size;
        drop(config); 
        
        if should_export {
            let state_clone = state.clone();
            tokio::spawn(perform_export(state_clone));
        }
    }

    // 6. Notify TUI
    let _ = state.tx_notify.send(());

    // 7. Smash into Physics Engine
    let collider = state.collider.clone();
    let text = serde_json::to_string(&parsed.payload).unwrap_or_default();
    // Hash conversation ID or standard
    let conversation_hash = {
         use std::hash::{Hash, Hasher};
         let mut hasher = std::collections::hash_map::DefaultHasher::new();
         // Use a consistent ID if present, else random or based on time
         "default_channel".hash(&mut hasher); 
         hasher.finish()
    };
    
    let type_name_clone = type_name.clone();

    // Extract image_path if present in payload
    let image_path = parsed.payload.get("image_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mut event = ConversationEvent {
        text,
        conversation_hash,
        actor_type: 1, // Human/System
        image_path,
        ..Default::default()
    };

    // Offload to blocking thread if needed, or just run (it sends to background via ONNX internal threads maybe?)
    // But `embed` is blocking. So `spawn_blocking` is best.
    tokio::task::spawn_blocking(move || {
        if let Err(e) = collider.smash(&mut event) {
            error!("Physics Engine Smash Error: {}", e);
        }

        // 7.5 GLiNER NER (if enabled)
        #[cfg(feature = "oar")]
        {
             let mut c = collider.inner.lock();
             if let Ok(entities) = c.process_ner(&event.text) {
                 if !entities.is_empty() {
                     info!("🧬 NER Extracted {} entities", entities.len());
                     for entity in entities {
                         info!("   - {}: {} ({:.2})", entity.label, entity.text, entity.score);
                     }
                 }
             }
        }

        // 8. Structured Extraction if OCR happened
        let type_name_target = type_name_clone.clone();
        let text_target = event.text.clone();
        let image_filename = event.image_path.as_ref()
            .map(|p| std::path::Path::new(p).file_name().unwrap_or_default().to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        

        
        // 8. Structured Extraction if OCR happened (and model available)
        let type_name_target = type_name_clone.clone();
        let text_target = event.text.clone();
        
        {
            let mut collider_guard = state.collider.inner.lock();
            if let Some(model) = &mut collider_guard.gliner_model { 
                 if let Some(json) = Some(extract_structured_data(&text_target, model)) {
                     info!("✅ Structured Data Extracted for {}: {}", type_name_target, json);
                 }
            }
        }
    });

    // 8. Return Success
    (StatusCode::OK, Json(serde_json::json!({"status": "received", "event_type": type_name}))).into_response()
}

async fn config_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let mut config = state.config.write();
    let mut updated = false;

    if let Some(url) = input.get("webhook_url").filter(|s| !s.is_empty()) {
        config.webhook_url = Some(url.clone());
        info!("🔗 Webhook updated via API: {}", url);
        updated = true;
    }

    if let Some(iri) = input.get("ontology_iri").filter(|s| !s.is_empty()) {
        config.ontology_iri = iri.clone();
        info!("🧠 Ontology IRI updated via API: {}", iri);
        updated = true;
    }

    if let Some(batch) = input.get("batch_size").and_then(|s| s.parse().ok()) {
        config.export_batch_size = batch;
        info!("📦 Batch size updated: {}", batch);
        updated = true;
    }

    Json(serde_json::json!({
        "status": if updated { "updated" } else { "unchanged" },
        "config": {
            "webhook_url": config.webhook_url,
            "ontology_iri": config.ontology_iri,
            "export_batch_size": config.export_batch_size
        }
    }))
}

async fn list_schemas_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let schemas = state.schemas.read();
    let list: Vec<_> = schemas.values().map(|s| {
        serde_json::json!({
            "name": s.name,
            "fields": s.fields,
            "sample": s.sample_data,
            "context": s.jsonld_context
        })
    }).collect();
    Json(serde_json::json!({ "schemas": list }))
}

async fn get_schema_handler(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let schemas = state.schemas.read();
    if let Some(schema) = schemas.get(&name) {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "name": schema.name,
                "fields": schema.fields,
                "sample": schema.sample_data,
                "context": schema.jsonld_context
            }))
        ).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Schema not found" }))
        ).into_response()
    }
}

async fn list_events_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let db = state.db.lock();
    let mut stmt = db.prepare("SELECT type, payload, received_at FROM events ORDER BY received_at DESC LIMIT 50").unwrap();
    let events_iter = stmt.query_map([], |row| {
        Ok(serde_json::json!({
            "type": row.get::<_, String>(0)?,
            "payload": row.get::<_, String>(1)?,
            "received_at": row.get::<_, String>(2)?,
        }))
    }).unwrap();

    let events: Vec<_> = events_iter.map(|r| r.unwrap()).collect();
    Json(serde_json::json!({ "events": events }))
}

async fn health_handler() -> impl IntoResponse {
    (axum::http::StatusCode::OK, "OK")
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {

    let m = state.metrics.lock();
    let q = state.export_queue.lock();
    let config = state.config.read();
    
    let json = serde_json::json!({
        "events_ingested": m.events_ingested,
        "schemas_learned": m.schemas_learned,
        "exports_attempted": m.exports_attempted,
        "exports_succeeded": m.exports_succeeded,
        "exports_failed": m.exports_failed,
        "last_export_at": m.last_export_at,
        "pending_exports": q.len(),
        "webhook_configured": config.webhook_url.is_some(),
        "export_batch_size": config.export_batch_size,
        "physics": {
            "frames_written": state.collider.metrics().frames_written.get(),
            "buffer_head": state.collider.metrics().buffer_head.get(),
        }
    });
    
    axum::Json(json)
}


// =================================================================================
// 5. THE AGENT (The Chat Logic)
// =================================================================================

#[derive(Deserialize)]
struct ChatInput { msg: String }

#[derive(Debug, Deserialize)]
struct LLMResponse {
    intent: Option<String>,
    #[serde(default)]
    slots: Vec<Slot>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    structured_data: Option<serde_json::Value>,
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
    // LM Studio OpenAI-compatible API
    let base_url = env::var("LLM_BASE_URL").unwrap_or_else(|_| "http://192.168.0.141:1234/v1".to_string());
    
    // Prepare Client
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", base_url);

    // Construct OpenAI-compatible payload
    // Note: LM Studio may not support response_format, so we instruct in prompt
    let payload = serde_json::json!({
        "model": "local-model",
        "messages": [
            { "role": "system", "content": format!("{}\n\nIMPORTANT: Always respond with valid JSON only, no markdown code blocks.", system_prompt) },
            { "role": "user", "content": input }
        ],
        "temperature": 0.7
    });

    // Send Request
    let res = client.post(&url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !res.status().is_success() {
        return Err(format!("LLM Error: {:?}", res.text().await));
    }

    // Parse Response (OpenAI format)
    let body: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
    
    // Extract from: choices[0].message.content
    let text_content = body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("Invalid LLM response structure")?;

    info!("🤖 Raw LLM Content: {}", text_content);

    // Strip thinking chain (e.g., <think>...</think>) from Qwen and similar models
    let cleaned = strip_thinking_chain(text_content);
    info!("🤖 Cleaned LLM JSON: {}", cleaned);

    // Parse the JSON string
    match serde_json::from_str::<LLMResponse>(&cleaned) {
        Ok(res) if res.intent.is_some() || res.message.is_some() || !res.slots.is_empty() => {
             Ok(res)
        },
        _ => {
            // Fallback: If it's valid JSON but not a full LLMResponse, wrap it
            match serde_json::from_str::<serde_json::Value>(&cleaned) {
                Ok(val) => Ok(LLMResponse {
                    intent: None,
                    slots: Vec::new(),
                    message: None,
                    structured_data: Some(val),
                }),
                Err(e) => Err(format!("Failed to parse JSON from LLM: {}. Content: {}", e, cleaned))
            }
        }
    }
}

/// Strip <think>...</think> tags and extract JSON from LLM response
fn strip_thinking_chain(content: &str) -> String {
    let mut result = content.to_string();
    
    // Remove <think>...</think> blocks (case insensitive, handles multiline)
    if let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        }
    }
    
    // Try to find JSON object in remaining text
    let trimmed = result.trim();
    if let Some(json_start) = trimmed.find('{') {
        if let Some(json_end) = trimmed.rfind('}') {
            return trimmed[json_start..=json_end].to_string();
        }
    }
    
    trimmed.to_string()
}

/// Extract structured JSON from OCR content using the local LLM
/// Extract structured JSON using GLiNER v2
fn extract_structured_data(text: &str, ner_model: &mut GlinerModel) -> serde_json::Value {
    // Define the schema (prompts) we want to extract
    // For invoices, especially Portuguese ones
    let schema = vec![
        "Invoice Number", "Número da Nota",
        "Total Amount", "Valor Total", 
        "Date", "Data de Emissão",
        "Vendor", "Emitente",
        "Receiver", "Destinatário",
        "CNPJ",
        "Item Description", "Descrição",
        "Tax", "Imposto"
    ];

    match ner_model.predict_entities(text, &schema, 0.4) {
        Ok(entities) => {
            // Group by label
            let mut map = serde_json::Map::new();
            for entity in entities {
                // Take the first occurrence or list? For simple KV, list might be better or last.
                // Let's use a list for repeated fields.
                let key = entity.label;
                let val = serde_json::Value::String(entity.text);
                
                map.entry(key)
                    .or_insert_with(|| serde_json::Value::Array(Vec::new()))
                    .as_array_mut()
                    .unwrap()
                    .push(val);
            }
            serde_json::Value::Object(map)
        },
        Err(e) => {
            eprintln!("GLiNER extraction failed: {}", e);
            serde_json::json!({"error": e.to_string()})
        }
    }
}

#[derive(Serialize)]
struct ChatResponse {
    role: String,
    content: String,
    intent: Option<String>,
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ChatInput>,
) -> Json<ChatResponse> {
    // 1. Prepare Context (Hold Locks briefly)
    let system_prompt = {
        let schemas = state.schemas.read();
        let mut session = state.session.lock();
        
        // Update history instantly (Escape User Input)
        session.history.push(("User".into(), encode_minimal(&input.msg)));

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
    let (response_text, intent) = {
         let schemas = state.schemas.read();
         let mut session = state.session.lock();

         match llm_result {
            Ok(llm_res) => {
                // Update Intent
                if let Some(new_intent) = llm_res.intent.clone() {
                    if schemas.contains_key(&new_intent) {
                        session.active_intent = Some(new_intent);
                    }
                }

                // Merge Slots
                for slot in llm_res.slots {
                    session.collected_slots.insert(slot.name, slot.value);
                }

                // Gap Analysis
                let mut final_msg = llm_res.message.unwrap_or_else(|| "Ok.".to_string());
                if let Some(intent) = &session.active_intent {
                    if let Some(schema) = schemas.get(intent) {
                        let missing = check_missing_slots(schema, &session.collected_slots);
                        
                        if missing.is_empty() {
                             final_msg.push_str(&format!("<div class='mt-3 p-2.5 rounded-lg bg-success-500/10 border border-success-500/20 text-success-400 text-[11px]'>✓ Complete! Ready to execute <strong>{}</strong></div>", intent));
                             // Persist to DB if we wanted to...
                             session.active_intent = None;
                             session.collected_slots.clear();
                        } else {
                             final_msg.push_str(&format!("<div class='mt-3 p-2.5 rounded-lg bg-warning-400/10 border border-warning-400/20 text-warning-400 text-[11px]'>Missing: {}</div>", missing.join(", ")));
                        }
                    }
                }
                session.history.push(("Agent".into(), final_msg.clone()));
                (final_msg, session.active_intent.clone())
            },
            Err(e) => {
                let msg = format!("<span class='text-red-400'>Error: {}</span>", encode_minimal(&e));
                session.history.push(("System".into(), msg.clone()));
                (msg, None)
            }
        }
    };

    Json(ChatResponse {
        role: "assistant".into(),
        content: response_text,
        intent,
    })
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
        let mut schema = LearnedSchema::new("Test".into(), "{}".into(), None);
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