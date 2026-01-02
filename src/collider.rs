use memmap2::MmapMut;
use ndarray::Array2;
use ort::session::{Session};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use prometheus::{Counter, Histogram, IntGauge};
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Instant};

// Use /tmp on macOS for file-backed shared memory simulation
#[cfg(target_os = "macos")]
const SHM_PATH: &str = "/tmp/cs_physics";
#[cfg(not(target_os = "macos"))]
const SHM_PATH: &str = "/dev/shm/cs_physics";

const BUFFER_SIZE: usize = 100_000;
const HEADER_SIZE: usize = 64;
const MAGIC: u64 = 0x50485953_49435300; // "PHYSICS\0"

/// Cache-line aligned frame for optimal memory access
#[repr(C, align(64))]
#[derive(Debug)]
pub struct ParticleFrame {
    // ═══════════════════════════════════════════════════════════
    // SYNCHRONIZATION (16 bytes)
    // ═══════════════════════════════════════════════════════════
    /// SeqLock sequence number. Odd = write in progress, Even = valid
    pub sequence: AtomicU64,
    /// Monotonic frame ID (for ordering and gap detection)
    pub frame_id: u64,

    // ═══════════════════════════════════════════════════════════
    // SEMANTIC MASS (4096 bytes) - bge-m3 dense embedding
    // ═══════════════════════════════════════════════════════════
    pub semantic: [f32; 1024],

    // ═══════════════════════════════════════════════════════════
    // KINETIC STATE (32 bytes)
    // ═══════════════════════════════════════════════════════════
    /// Log-seconds since previous event in this conversation
    pub delta_time: f32,
    /// Processing/execution duration in milliseconds
    pub duration_ms: f32,
    /// Throughput: tokens/sec for LLM, messages/min for human
    pub velocity: f32,
    /// Overlap coefficient: 1.0 if interrupted previous, 0.0 otherwise
    pub interrupt: f32,
    /// Timestamp (Unix epoch microseconds)
    pub timestamp_us: i64,
    /// Conversation ID hash (for multi-tenant separation)
    pub conversation_hash: u64,

    // ═══════════════════════════════════════════════════════════
    // POSITIONAL STATE (32 bytes)
    // ═══════════════════════════════════════════════════════════
    /// Current workflow node ID (-1 if not in workflow)
    pub workflow_node: i32,
    /// Case status enum (0=new, 1=open, 2=pending, 3=resolved, 4=closed)
    pub case_status: i32,
    /// Lead status enum (0=cold, 1=warm, 2=hot, 3=converted, 4=lost)
    pub lead_status: i32,
    /// Data completeness ratio [0.0, 1.0]
    pub fill_rate: f32,
    /// Actor type (0=system, 1=human, 2=agent, 3=tool)
    pub actor_type: i32,
    /// Intent classification ID
    pub intent_id: i32,
    /// Sentiment score [-1.0, 1.0]
    pub sentiment: f32,
    /// Confidence of classification [0.0, 1.0]
    pub confidence: f32,

    // ═══════════════════════════════════════════════════════════
    // SPIN STATE (16 bytes) - Orientation quaternion
    // ═══════════════════════════════════════════════════════════
    /// Quaternion (w, x, y, z) representing semantic orientation
    /// Derived from topic/intent clustering
    pub spin: [f32; 4],

    // ═══════════════════════════════════════════════════════════
    // PADDING (32 bytes) - Future expansion + cache alignment
    // ═══════════════════════════════════════════════════════════
    pub _reserved: [u8; 32],
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<ParticleFrame>() == 4224);
const _: () = assert!(std::mem::align_of::<ParticleFrame>() == 64);

/// Metrics for observability
#[derive(Clone, Debug)]
pub struct ColliderMetrics {
    pub frames_written: Counter,
    pub write_latency: Histogram,
    pub inference_latency: Histogram,
    pub buffer_head: IntGauge,
    pub sequence_gaps: Counter,
}

impl ColliderMetrics {
    pub fn new() -> Self {
        Self {
            frames_written: Counter::new("collider_frames_written_total", "Total frames written").unwrap(),
            write_latency: Histogram::with_opts(
                prometheus::HistogramOpts::new("collider_write_latency_seconds", "Write latency")
                    .buckets(vec![0.000001, 0.00001, 0.0001, 0.001, 0.01])
            ).unwrap(),
            inference_latency: Histogram::with_opts(
                prometheus::HistogramOpts::new("collider_inference_latency_seconds", "ONNX inference latency")
                    .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
            ).unwrap(),
            buffer_head: IntGauge::new("collider_buffer_head", "Current write position").unwrap(),
            sequence_gaps: Counter::new("collider_sequence_gaps_total", "Detected sequence gaps").unwrap(),
        }
    }
}

pub struct Collider {
    mmap: MmapMut,
    cursor: usize,
    frame_counter: u64,
    model: Option<Session>,
    tokenizer: Option<tokenizers::Tokenizer>,
    pub metrics: ColliderMetrics,
    last_timestamps: std::collections::HashMap<u64, i64>, // conversation_hash -> last timestamp
}

// Safe to implement Debug manually or derive if all fields are Debug. 
// Session and Tokenizer might not be Debug. 
// Let's implement Debug manually to be safe and cleaner.
impl std::fmt::Debug for Collider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collider")
         .field("cursor", &self.cursor)
         .field("frame_counter", &self.frame_counter)
         .field("metrics", &self.metrics)
         .finish()
    }
}

unsafe impl Send for Collider {}
unsafe impl Sync for Collider {}

impl Collider {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Calculate total size
        let total_size = HEADER_SIZE + (BUFFER_SIZE * std::mem::size_of::<ParticleFrame>());

        // Create/open shared memory
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(SHM_PATH)?;
        file.set_len(total_size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize header
        let header = mmap.as_mut_ptr();
        unsafe {
            std::ptr::write(header as *mut u64, MAGIC);
            std::ptr::write(header.add(8) as *mut u32, 1); // version
            std::ptr::write(header.add(12) as *mut u32, std::mem::size_of::<ParticleFrame>() as u32);
            std::ptr::write(header.add(16) as *mut u64, BUFFER_SIZE as u64);
            // head and tail start at 0
        }

        // Load ONNX model with optimization
        // Wrap in catch_unwind because ort might panic if libonnxruntime is missing
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match Session::builder().and_then(|b| 
                b.with_optimization_level(GraphOptimizationLevel::Level3)
                 .and_then(|b| b.with_intra_threads(4))
                 .and_then(|b| b.commit_from_file(model_path))
            ) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!("⚠️ WARN: Failed to load ONNX model ({}). Running in MOCK mode.", e);
                    None
                }
            }
        })).unwrap_or_else(|_| {
            eprintln!("⚠️ CRITICAL: ORT Runtime Panic (likely missing libonnxruntime). Defaulting to MOCK mode.");
            None
        });

        let tokenizer = match tokenizers::Tokenizer::from_file(tokenizer_path) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("⚠️ WARN: Failed to load Tokenizer ({}). Running in MOCK mode.", e);
                None
            }
        };

        Ok(Self {
            mmap,
            cursor: 0,
            frame_counter: 0,
            model,
            tokenizer,
            metrics: ColliderMetrics::new(),
            last_timestamps: std::collections::HashMap::new(),
        })
    }

    /// Compute embedding via LM Studio OpenAI-compatible API
    fn embed(&mut self, text: &str) -> Result<[f32; 1024], Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        let base_url = std::env::var("LLM_BASE_URL")
            .unwrap_or_else(|_| "http://192.168.0.141:1234/v1".to_string());
        let url = format!("{}/embeddings", base_url);
        
        // Use blocking reqwest since we're in a sync context
        let client = reqwest::blocking::Client::new();
        let payload = serde_json::json!({
            "model": "text-embedding-bge-m3@f16",
            "input": text
        });
        
        let res = client.post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send();
            
        match res {
            Ok(response) => {
                if response.status().is_success() {
                    let body: serde_json::Value = response.json()?;
                    // OpenAI embeddings format: data[0].embedding
                    if let Some(embedding_array) = body["data"][0]["embedding"].as_array() {
                        let mut embedding = [0.0f32; 1024];
                        for (i, val) in embedding_array.iter().take(1024).enumerate() {
                            embedding[i] = val.as_f64().unwrap_or(0.0) as f32;
                        }
                        self.metrics.inference_latency.observe(start.elapsed().as_secs_f64());
                        return Ok(embedding);
                    }
                }
                eprintln!("⚠️ LM Studio embedding failed, using zero vector");
            }
            Err(e) => {
                eprintln!("⚠️ LM Studio connection error: {}", e);
            }
        }
        
        // Fallback to zero embedding
        Ok([0.0; 1024])
    }

    /// Write a frame to the ring buffer using SeqLock protocol
    pub fn smash(&mut self, event: &ConversationEvent) -> Result<(), Box<dyn std::error::Error>> {
        let write_start = Instant::now();

        // 1. Compute embedding
        // Handle potential empty strings or tokenizer/model failures gracefully
        let semantic = self.embed(&event.text).unwrap_or([0.0; 1024]);

        // 2. Calculate kinetics
        let now = std::time::SystemTime::now();
        let now_us = now.duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::ZERO)
            .as_micros() as i64;

        let delta_time = self.last_timestamps
            .get(&event.conversation_hash)
            .map(|&last| {
                let diff_us = now_us - last;
                if diff_us > 0 {
                    ((diff_us as f64) / 1_000_000.0).ln() as f32
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);

        self.last_timestamps.insert(event.conversation_hash, now_us);

        // 3. Calculate buffer offset
        let frame_offset = HEADER_SIZE + (self.cursor * std::mem::size_of::<ParticleFrame>());
        
        // Unsafe pointer arithmetic to get mutable reference to the frame
        // This is safe because we own the mmap and are the single writer
        let frame_ptr = unsafe { self.mmap.as_mut_ptr().add(frame_offset) as *mut ParticleFrame };

        // 4. SeqLock write protocol
        unsafe {
            let frame = &mut *frame_ptr;

            // 4a. Increment sequence to ODD (write in progress)
            let old_seq = frame.sequence.load(Ordering::Relaxed);
            let new_seq = old_seq.wrapping_add(1);
            frame.sequence.store(new_seq, Ordering::Release);

            // 4b. Memory barrier
            std::sync::atomic::fence(Ordering::Release);

            // 4c. Write payload (non-atomic fields)
            frame.frame_id = self.frame_counter;
            frame.semantic = semantic;
            frame.delta_time = delta_time;
            frame.duration_ms = event.duration_ms;
            frame.velocity = event.velocity;
            frame.interrupt = event.interrupt;
            frame.timestamp_us = now_us;
            frame.conversation_hash = event.conversation_hash;
            frame.workflow_node = event.workflow_node;
            frame.case_status = event.case_status;
            frame.lead_status = event.lead_status;
            frame.fill_rate = event.fill_rate;
            frame.actor_type = event.actor_type;
            frame.intent_id = event.intent_id;
            frame.sentiment = event.sentiment;
            frame.confidence = event.confidence;
            frame.spin = event.spin;

            // 4d. Memory barrier
            std::sync::atomic::fence(Ordering::Release);

            // 4e. Increment sequence to EVEN (write complete)
            frame.sequence.store(new_seq.wrapping_add(1), Ordering::Release);
        }

        // 5. Update head pointer atomically
        let head_ptr = unsafe { (self.mmap.as_ptr().add(24) as *const AtomicU64).as_ref().unwrap() };
        head_ptr.store(self.cursor as u64, Ordering::Release);

        // 6. Advance cursor
        self.cursor = (self.cursor + 1) % BUFFER_SIZE;
        self.frame_counter += 1;

        // 7. Metrics
        self.metrics.frames_written.inc();
        self.metrics.buffer_head.set(self.cursor as i64);
        self.metrics.write_latency.observe(write_start.elapsed().as_secs_f64());

        Ok(())
    }
}

/// Input event structure
#[derive(Debug, Clone)]
pub struct ConversationEvent {
    pub text: String,
    pub conversation_hash: u64,
    pub duration_ms: f32,
    pub velocity: f32,
    pub interrupt: f32,
    pub workflow_node: i32,
    pub case_status: i32,
    pub lead_status: i32,
    pub fill_rate: f32,
    pub actor_type: i32,
    pub intent_id: i32,
    pub sentiment: f32,
    pub confidence: f32,
    pub spin: [f32; 4],
}

impl Default for ConversationEvent {
    fn default() -> Self {
        Self {
            text: String::new(),
            conversation_hash: 0,
            duration_ms: 0.0,
            velocity: 0.0,
            interrupt: 0.0,
            workflow_node: -1,
            case_status: 0,
            lead_status: 0,
            fill_rate: 0.0,
            actor_type: 0,
            intent_id: -1,
            sentiment: 0.0,
            confidence: 0.0,
            spin: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
        }
    }
}

use std::sync::Arc;
use parking_lot::Mutex;

/// Thread-safe collider for use in Axum handlers
#[derive(Debug)]
pub struct SharedCollider {
    pub inner: Mutex<Collider>,
}

impl SharedCollider {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        Ok(Arc::new(Self {
            inner: Mutex::new(Collider::new(model_path, tokenizer_path)?),
        }))
    }

    pub fn smash(&self, event: &ConversationEvent) -> Result<(), Box<dyn std::error::Error>> {
        self.inner.lock().smash(event)
    }

    pub fn metrics(&self) -> ColliderMetrics {
        self.inner.lock().metrics.clone()
    }
}

