// use image::GenericImageView;
use memmap2::MmapMut;
// use ndarray::Array2;
use tracing::info;
use prometheus::{Counter, Histogram, IntGauge};
use std::fs::OpenOptions;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::Instant;

const BUFFER_SIZE: usize = 100_000;

#[repr(C, align(64))]
#[derive(Debug)]
pub struct Header {
    pub magic: u64,
    pub version: u32,
    pub frame_size: u32,
    pub capacity: usize,
    pub head: AtomicU64,
    pub tail: AtomicU64,
    pub _pad: [u64; 2],
}

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
    pub _sequence_gaps: Counter,
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
            _sequence_gaps: Counter::new("collider_sequence_gaps_total", "Detected sequence gaps").unwrap(),
        }
    }
}

use ort::session::{Session, builder::GraphOptimizationLevel};

use ort::value::Value;
use tokenizers::Tokenizer;

// ... imports ...

pub struct Collider {
    shared: SharedCollider,
    cursor: usize,
    frame_counter: u64,
    pub metrics: ColliderMetrics,
    last_timestamps: std::collections::HashMap<u64, i64>,
    
    // ONNX Sessions
    embedding_session: Option<Session>, 
    tokenizer: Option<Tokenizer>,
    #[cfg(feature = "oar")]
    pub ocr_pipeline: Option<Box<dyn crate::ocr_pipeline::OcrPipelineTrait>>,
    pub gliner_model: Option<crate::gliner::GlinerModel>,
}

impl std::fmt::Debug for Collider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collider")
         .field("cursor", &self.cursor)
         .field("frame_counter", &self.frame_counter)
         .field("metrics", &self.metrics)
         .field("metrics", &self.metrics)
         .finish()
    }
}

unsafe impl Send for Collider {}
unsafe impl Sync for Collider {}

// --- Shared Memory Layout V2 ---
// [ Header (64b) ] [ Control Plane (1MB) ] [ Ring Buffer (Frames...) ]

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Attractor {
    pub position: [f32; 1024], // Target embedding
    pub mass: f32,             // Influence strength
    pub label: [u8; 32],       // Short tag (ASCII)
    pub padding: [u8; 60],     // Align to cache line (approx) -> 4096 + 4 + 32 + 60 = 4192 bytes
}

#[repr(C)]
#[derive(Debug)]
pub struct ControlRegion {
    pub global_gravity: AtomicU32, // f32 cast to u32 for atomic access
    pub friction: AtomicU32,       // f32 cast to u32
    pub attractor_count: AtomicU32,
    pub _pad1: u32,
    pub attractors: [Attractor; 64], // Fixed capacity for now
}

// Total Control Region size: 16 (header) + 64 * 4192 = ~268KB. We reserve 1MB.
const CONTROL_PLANE_SIZE: usize = 1024 * 1024; // 1MB

pub struct SharedCollider {
    _file: std::fs::File,
    _mmap: MmapMut,
    pub header: *mut Header,
    pub control: *mut ControlRegion, // Pointer to Control Plane
    pub frames: *mut ParticleFrame,
    pub capacity: usize,
    // ...
}

unsafe impl Send for SharedCollider {}
unsafe impl Sync for SharedCollider {}

impl SharedCollider {
    pub fn new(capacity: usize) -> std::io::Result<Self> {
        let path = if cfg!(target_os = "macos") {
            "/tmp/cs_physics"
        } else {
            "/dev/shm/cs_physics"
        };

        // Calculate V2 Layout
        let header_size = std::mem::size_of::<Header>();
        // Align Control Plane to 4KB page
        let control_offset = (header_size + 4095) & !4095;
        
        let frame_size = std::mem::size_of::<ParticleFrame>();
        
        // Ring Buffer starts after Control Plane (also aligned)
        let ring_offset = control_offset + CONTROL_PLANE_SIZE;
        
        // Total size
        let total_size = ring_offset + (frame_size * capacity);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(total_size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize Header if fresh
        let header_ptr = mmap.as_mut_ptr() as *mut Header;
        unsafe {
            if (*header_ptr).magic != 0xCAFEBABE {
                (*header_ptr).magic = 0xCAFEBABE;
                (*header_ptr).version = 2; // V2
                (*header_ptr).capacity = capacity;
                (*header_ptr).frame_size = frame_size as u32;
                (*header_ptr).head.store(0, Ordering::SeqCst);
                (*header_ptr).tail.store(0, Ordering::SeqCst);
                
                // Init Control Region
                let control_ptr = mmap.as_mut_ptr().add(control_offset) as *mut ControlRegion;
                std::ptr::write_bytes(control_ptr, 0, 1);
                // Default Gravity parameters (using u32 bits of f32 1.0)
                let f1: f32 = 1.0;
                let f01: f32 = 0.1;
                (*control_ptr).global_gravity.store(f1.to_bits(), Ordering::Relaxed);
                (*control_ptr).friction.store(f01.to_bits(), Ordering::Relaxed);
                (*control_ptr).attractor_count.store(0, Ordering::Relaxed);
            }
        }
        
        let control_ptr = unsafe { mmap.as_mut_ptr().add(control_offset) as *mut ControlRegion };
        let frames_ptr = unsafe { mmap.as_mut_ptr().add(ring_offset) as *mut ParticleFrame };

        Ok(Self {
            _file: file,
            _mmap: mmap,
            header: header_ptr,
            control: control_ptr,
            frames: frames_ptr,
            capacity,
        })
    }
}

impl Collider {
    pub fn new(
        #[cfg(feature = "oar")]
        ocr_pipeline: Option<Box<dyn crate::ocr_pipeline::OcrPipelineTrait>>, 
        gliner_model: Option<crate::gliner::GlinerModel>,
        embedding_model_path: Option<&str>,
        tokenizer_path: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize Shared Memory
        let shared = SharedCollider::new(BUFFER_SIZE)?;

        // Load Embedding Model (BGE-M3-Int8)
        let mut embedding_session = None;
        let mut tokenizer = None;

        if let (Some(model_path), Some(tok_path)) = (embedding_model_path, tokenizer_path) {
            info!("🧠 Loading Embedding Model from: {}", model_path);
            
            match Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(model_path) {
                    Ok(session) => {
                         embedding_session = Some(session);
                         
                         info!("🔤 Loading Tokenizer from: {}", tok_path);
                         match Tokenizer::from_file(tok_path) {
                             Ok(t) => tokenizer = Some(t),
                             Err(e) => eprintln!("⚠️ Failed to load tokenizer: {}", e),
                         }
                    },
                    Err(e) => eprintln!("⚠️ Failed to load embedding session: {}", e),
                }
        }

        Ok(Self {
            shared,
            cursor: 0,
            frame_counter: 0,
            metrics: ColliderMetrics::new(),
            last_timestamps: std::collections::HashMap::new(),
            embedding_session,
            tokenizer,
            #[cfg(feature = "oar")]
            ocr_pipeline,
            gliner_model,
        })
    }

    /// Run full OCR Pipeline on a file path
    pub fn process_ocr_image(&mut self, path: &str) -> Result<String, Box<dyn std::error::Error>> {
        #[cfg(feature = "oar")]
        if let Some(pipeline) = &mut self.ocr_pipeline {
             info!("🧪 Using OCR Pipeline: {}", pipeline.name());
             let result = pipeline.process_file(path)?;
             return Ok(result.markdown);
        }
        Ok(String::new())
    }

    pub fn process_ner(&mut self, text: &str) -> Result<Vec<crate::gliner::Entity>, Box<dyn std::error::Error>> {
        if let Some(model) = &mut self.gliner_model {
            // Default labels for general entity extraction
            let labels = ["person", "organization", "date", "money", "location"];
            return model.predict_entities(text, &labels, 0.4).map_err(|e| e.into());
        }
        Ok(Vec::new())
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Compute embedding with Deep Zero-Copy using ort::IoBinding
    fn embed(&mut self, text: &str, output: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();

        // 1. Local BGE-M3 Inference (Deep Zero-Copy attempt via IoBinding)
        if let (Some(session), Some(tokenizer)) = (&mut self.embedding_session, &self.tokenizer) {
             let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
             println!("Model Outputs: {:?}", output_names); // Debug
             let encoding = tokenizer.encode(text, true)
                .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Tokenizer error: {}", e))) as Box<dyn std::error::Error>)?;
             let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
             let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
             
             let batch_size = 1;
             let seq_len = input_ids.len();
             
             let input_tensor = ndarray::Array2::from_shape_vec((batch_size, seq_len), input_ids)?;
             let attention_tensor = ndarray::Array2::from_shape_vec((batch_size, seq_len), attention_mask)?;
             
             // Create IOBinding
             let mut binding = session.create_binding()?;
             
             // Bind Inputs (Consumes tensor, zero-copy to ort Value)
             let input_val = Value::from_array(input_tensor)?;
             let attention_val = Value::from_array(attention_tensor)?;
             
             binding.bind_input("input_ids", &input_val)?;
             binding.bind_input("attention_mask", &attention_val)?;
             
             // Output: Runtime allocates
             // We use One-Copy Output strategy for stability (4KB copy is negligible)
             let output_name = "last_hidden_state";
             let mem_info = session.allocator().memory_info();
             binding.bind_output_to_device(output_name, &mem_info)?;
             
             // Run
             let outputs = session.run_binding(&binding)?;
             
             // Extract and copy
             // try_extract_tensor returns (shape, data_slice)
             let (_, data) = outputs[output_name].try_extract_tensor::<f32>()?;
             
             // Copy to shared memory
             for (i, v) in data.iter().take(1024).enumerate() {
                 output[i] = *v;
             }
             
             self.metrics.inference_latency.observe(start.elapsed().as_secs_f64());
             return Ok(());
        }

        // 2. Fallback to HTTP (Legacy / Debug)
        // ... (previous logic, but writing to output slice)
        
        let base_url = std::env::var("LLM_BASE_URL")
            .unwrap_or_else(|_| "http://192.168.0.141:1234/v1".to_string());
        let url = format!("{}/embeddings", base_url);
        
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
                    if let Some(embedding_array) = body["data"][0]["embedding"].as_array() {
                        for (i, val) in embedding_array.iter().take(1024).enumerate() {
                            output[i] = val.as_f64().unwrap_or(0.0) as f32;
                        }
                        self.metrics.inference_latency.observe(start.elapsed().as_secs_f64());
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                eprintln!("⚠️ LM Studio connection error: {}", e);
            }
        }
        
        // Zero out on failure
        output.fill(0.0);
        Ok(())
    }

    /// Write a frame to the ring buffer using SeqLock protocol
    /// Applies Gravitational Learning logic
    pub fn smash(&mut self, event: &mut ConversationEvent) -> Result<(), Box<dyn std::error::Error>> {
        let write_start = Instant::now();

        // 0. OCR Enrichment (if image path provided)
        if let Some(path) = &event.image_path {
             // Unified Reconstruction Pipeline (ocr + layout + merge)
             match self.process_ocr_image(path) {
                 Ok(extracted_text) => {
                     if !extracted_text.is_empty() {
                         // println!("📄 Markdown Extracted: {:.50}...", extracted_text); // debug
                         event.text.push_str("\n");
                         event.text.push_str(&extracted_text);
                     }
                 }
                 Err(e) => eprintln!("⚠️ OCR/Reconstruction Failed for {}: {}", path, e),
             }
        }

        // 5. Get pointer from SharedCollider (EARLY ACCESS for Zero-Copy)
        let frame_ptr = unsafe { self.shared.frames.add(self.cursor) };

        // 6. SeqLock write protocol
        unsafe {
            let frame = &mut *frame_ptr;

            let old_seq = frame.sequence.load(Ordering::Relaxed);
            let new_seq = old_seq.wrapping_add(1);
            frame.sequence.store(new_seq, Ordering::Release);

            std::sync::atomic::fence(Ordering::Release);

            // 1. Compute embedding (Directly into Shared Memory!)
            // We do this *inside* the write lock because we are modifying the data
            // This blocks readers slightly longer but avoids the copy.
            self.embed(&event.text, &mut frame.semantic).unwrap_or_else(|e| {
                eprintln!("Embedding failed: {}", e);
                // Zero out if failed
                frame.semantic.fill(0.0);
            });
            
            let semantic_ref = &frame.semantic; // Use for gravity calculation

            // 2. Gravitational Learning (Physics)
            let mut gravity_boost = 0.0;
            let control = &*self.shared.control;
            let g_bits = control.global_gravity.load(Ordering::Relaxed);
            let g = f32::from_bits(g_bits);
            let count = control.attractor_count.load(Ordering::Relaxed).min(64) as usize;
            
            for i in 0..count {
                let attractor = &control.attractors[i];
                let sim = Self::cosine_similarity(semantic_ref, &attractor.position);
                if sim > 0.8 {
                     gravity_boost += attractor.mass * sim * g;
                }
            }

            // 3. Calculate Kinetics (Restored)
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
            
            let final_velocity = event.velocity + gravity_boost;

            // 4. Write Frame Data
            frame.frame_id = self.frame_counter;
            // frame.semantic is already set!
            frame.delta_time = delta_time;
            frame.duration_ms = event.duration_ms;
            frame.velocity = final_velocity; 
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
            
            // TODO: Add field for attractor_label in ParticleFrame if we want to visualize it
            // For now, we just use the velocity boost effect

            std::sync::atomic::fence(Ordering::Release);

            frame.sequence.store(new_seq.wrapping_add(1), Ordering::Release);
        }

        // 7. Update head atomically
        unsafe {
            (*self.shared.header).head.store(self.cursor as u64, Ordering::Release);
        }

        // 8. Advance cursor
        self.cursor = (self.cursor + 1) % self.shared.capacity;
        self.frame_counter += 1;

        // 9. Metrics
        self.metrics.frames_written.inc();
        self.metrics.buffer_head.set(self.cursor as i64);
        self.metrics.write_latency.observe(write_start.elapsed().as_secs_f64());

        Ok(())
    }
}


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
    pub image_path: Option<String>,
    pub _structured_data: Option<serde_json::Value>,
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
            image_path: None,
            _structured_data: None,
        }
    }
}

use std::sync::Arc;
use parking_lot::Mutex;

/// Thread-safe collider for use in Axum handlers
#[derive(Debug)]
pub struct LockedCollider {
    pub inner: Mutex<Collider>,
}

impl LockedCollider {
    pub fn new(
        #[cfg(feature = "oar")]
        ocr_pipeline: Option<Box<dyn crate::ocr_pipeline::OcrPipelineTrait>>, 
        gliner_model: Option<crate::gliner::GlinerModel>,
        embedding_model_path: Option<&str>,
        tokenizer_path: Option<&str>
    ) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        Ok(Arc::new(Self {
            #[cfg(feature = "oar")]
            inner: Mutex::new(Collider::new(ocr_pipeline, gliner_model, embedding_model_path, tokenizer_path)?),
            #[cfg(not(feature = "oar"))]
            inner: Mutex::new(Collider::new(gliner_model, embedding_model_path, tokenizer_path)?),
        }))
    }

    pub fn smash(&self, event: &mut ConversationEvent) -> Result<(), Box<dyn std::error::Error>> {
        let mut collider = self.inner.lock();
        collider.smash(event)
    }

    pub fn metrics(&self) -> ColliderMetrics {
        self.inner.lock().metrics.clone()
    }
}

