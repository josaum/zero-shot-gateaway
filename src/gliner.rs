use anyhow::{Context, Result};
use ndarray::{Array2, Array3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
pub struct GlinerConfig {
    pub hidden_size: usize,
    pub max_width: usize,
    pub model_name: String,
    pub vocab_size: usize,
    pub class_token_index: Option<usize>,
    pub sep_token: Option<String>,
    #[serde(default)]
    pub words_splitter_type: Option<String>,
}

pub struct GlinerModel {
    session: Session,
    tokenizer: Tokenizer,
    config: GlinerConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct Entity {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub label: String,
    pub score: f32,
}

impl GlinerModel {
    pub fn new(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);

        // Load Tokenizer
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load Config
        let config_path = dir.join("gliner_config.json");
        let config_file = std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open config file: {:?}", config_path))?;
        let config: GlinerConfig = serde_json::from_reader(config_file)
            .with_context(|| "Failed to parse gliner_config.json")?;

        // Load Single Session (v2 Unified)
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(dir.join("model.onnx"))?;

        Ok(Self {
            session,
            tokenizer,
            config,
        })
    }

    pub fn predict_entities(&mut self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Entity>> {
        // 1. Tokenize Text
        let text_encoding = self.tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
        let text_ids: Vec<i64> = text_encoding.get_ids().iter().map(|&x| x as i64).collect();
        // text_ids from encode(..., true) usually implies [CLS] ... [SEP]
        
        let batch_size = 1;

        let mut input_ids = text_ids.clone();
        
        // Ensure text_ids ends with SEP?
        let sep_id_i64 = self.tokenizer.token_to_id(self.config.sep_token.as_deref().unwrap_or("<<SEP>>"))
             .or_else(|| self.tokenizer.token_to_id("[SEP]"))
             .unwrap_or(102) as i64;
        
        if let Some(&last) = input_ids.last() {
             if last != sep_id_i64 {
                 input_ids.push(sep_id_i64);
             }
        }
        
        // 2. Prepare Labels (Prompts)
        // Use class_token_index (e.g. 250103 for <<ENT>>)
        let class_token_id = self.config.class_token_index.unwrap_or(250103) as i64;

        let mut label_start_indices = Vec::new();
        let mut label_end_indices = Vec::new();

        for label in labels {
            let label_encoding = self.tokenizer.encode(*label, false).map_err(|e| anyhow::anyhow!(e))?;
            let label_ids: &[u32] = label_encoding.get_ids();
            
            // Push [ENT] token
            input_ids.push(class_token_id);

            label_start_indices.push(input_ids.len());
            // Convert to i64 and push
            for &id in label_ids {
                input_ids.push(id as i64);
            }
            label_end_indices.push(input_ids.len()); // End is exclusive? Python checks bounds.
        }

        // 3. Prepare Masks & Spans using WHITESPACE word splitting
        // The model was trained with words_splitter_type: "whitespace"
        // This means we split by whitespace FIRST, then map tokens to those words

        let seq_len = input_ids.len();
        let attention_mask_array = Array2::<i64>::ones((batch_size, seq_len));
        let mut words_mask_array = Array2::<i64>::zeros((batch_size, seq_len));

        let num_text_tokens = text_ids.len();

        // Step 1: Split original text into whitespace-delimited words and get their char spans
        let mut word_char_spans: Vec<(usize, usize)> = Vec::new();
        let mut pos = 0;
        for word in text.split_whitespace() {
            if let Some(start) = text[pos..].find(word) {
                let actual_start = pos + start;
                let actual_end = actual_start + word.len();
                word_char_spans.push((actual_start, actual_end));
                pos = actual_end;
            }
        }
        let num_words = word_char_spans.len();

        println!("DEBUG: Whitespace words count: {}", num_words);
        println!("DEBUG: Word char spans: {:?}", word_char_spans);

        // Step 2: Map each text token to its whitespace word ID (1-indexed, 0 = special tokens)
        let mut mapping: Vec<i64> = vec![0; seq_len];
        let tokens = text_encoding.get_tokens();
        let offsets = text_encoding.get_offsets();

        // CLS token (index 0) gets 0
        mapping[0] = 0;

        for token_idx in 1..num_text_tokens {
            let token = &tokens[token_idx];

            // Skip special tokens
            if token == "[SEP]" || token == "[CLS]" || token == "[PAD]" {
                mapping[token_idx] = 0;
                continue;
            }

            // Get token's character offsets
            if let Some(&(tok_start, tok_end)) = offsets.get(token_idx) {
                if tok_start == tok_end {
                    // Empty token (sometimes happens with special tokens)
                    mapping[token_idx] = 0;
                    continue;
                }

                // Find which whitespace word this token belongs to
                for (word_idx, &(word_start, word_end)) in word_char_spans.iter().enumerate() {
                    // Token belongs to word if token's start falls within word's span
                    if tok_start >= word_start && tok_start < word_end {
                        // Word IDs are 1-indexed (0 is reserved for special tokens)
                        mapping[token_idx] = (word_idx + 1) as i64;
                        break;
                    }
                }
            }
        }

        // Label tokens and padding get 0
        for i in num_text_tokens..seq_len {
            mapping[i] = 0;
        }

        // Apply to array
        for i in 0..seq_len {
            words_mask_array[[0, i]] = mapping[i];
        }

        let max_word_id = *mapping.iter().max().unwrap_or(&0);
        println!("DEBUG: Max word ID in mapping: {}", max_word_id);

        // 4. Generate Spans over whitespace words
        // Spans are [start, end] where both are word positions (0-indexed)
        // and end is INCLUSIVE. Width = end - start + 1.

        let max_width = self.config.max_width;
        let mut spans: Vec<[i64; 2]> = Vec::new();

        // Generate spans for each starting position and each width
        for start in 0..num_words {
            for width in 1..=max_width {
                let end = start + width - 1; // inclusive end
                if end < num_words {
                    spans.push([start as i64, end as i64]);
                } else {
                    // Padding span - still need consistent array size
                    spans.push([0, 0]);
                }
            }
        }

        println!("DEBUG: Generated {} spans for {} words", spans.len(), num_words);
    
        // Construct Span Arrays
        let num_spans = spans.len();
        let mut span_idx_array = Array3::<i64>::zeros((batch_size, num_spans, 2));
        let mut span_mask_array = Array2::<bool>::from_elem((batch_size, num_spans), false);

        for (i, span) in spans.iter().enumerate() {
            span_idx_array[[0, i, 0]] = span[0];
            span_idx_array[[0, i, 1]] = span[1];

            // Mark valid spans (not padding [0,0] spans that were added for out-of-bounds)
            let span_width = span[1] - span[0] + 1;
            // A span is valid if it's within bounds and has reasonable width
            let is_valid = span_width >= 1 && span_width <= max_width as i64;
            // Also check that it's not a padding span for an invalid position
            let span_end_in_bounds = (span[1] as usize) < num_words;
            span_mask_array[[0, i]] = is_valid && span_end_in_bounds;
        }

        // text_lengths should be the number of whitespace-split words
        let mut text_lengths_array = Array2::<i64>::zeros((batch_size, 1));
        text_lengths_array[[0, 0]] = num_words as i64; 
        
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)?;

        // Debug inputs before run
        println!("DEBUG: text_lengths = {}", num_words);
        println!("DEBUG: Span count: {}", num_spans);
        println!("DEBUG: Token mapping: {:?}", &mapping[..num_text_tokens.min(mapping.len())]);

        let inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids_array)?,
            "attention_mask" => Value::from_array(attention_mask_array)?,
            "words_mask" => Value::from_array(words_mask_array)?,
            "text_lengths" => Value::from_array(text_lengths_array)?,
            "span_idx" => Value::from_array(span_idx_array)?,
            "span_mask" => Value::from_array(span_mask_array)?,
        ];

        let outputs = self.session.run(inputs)?;
        
        // Debug outputs keys
        for (k, v) in outputs.iter() {
             println!("DEBUG: Output Key: {}, Type: {:?}", k, v.dtype());
        }

        let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?; // Shape: [batch, num_words, num_spans_per_word, num_classes] or similar
        
        // 6. Decode Output
        // ONNX Output Shape: [Batch, SequenceLength, NumSpansPerWord/K, NumClasses]
        // shape[0]=B, shape[1]=NumWords, shape[2]=K=12, shape[3]=NumClasses=3
        let num_detected_classes = shape[3] as usize; 
        
        // Validate class count?
        // if num_detected_classes != labels.len() { ... }

        let mut entities = Vec::new();
        
        // Total flattened spans = NumWords * K
        let total_spans = num_spans; // 288
        
        // Debug Tokens
        println!("DEBUG: Tokenizer output:");
        for (i, t) in text_encoding.get_tokens().iter().enumerate() {
            println!("  Token {}: {}  (Offsets: {:?})", i, t, text_encoding.get_offsets().get(i));
        }

        for i in 0..total_spans {
            for j in 0..labels.len() { // Iterate over classes
                if j >= num_detected_classes { break; }
                
                // Manual Indexing: [batch, word, span, class] flattened?
                // data is flat. 
                // data layout is C-contiguous: B -> Words -> K -> Classes
                // i corresponds to (Word * K + k) ?
                // Yes, spans were generated in that order.
                
                let idx = i * num_detected_classes + j;
                let score = if idx < data.len() { data[idx] } else { -100.0 };
                
                // Sigmoid
                let prob = 1.0 / (1.0 + (-score).exp());
                
                // Debug Scan: Check for "Cristiano Ronaldo" span [1, 2] (or [1, 3]?)
                let start_w = spans[i][0];
                let end_w = spans[i][1];
                if start_w == 1 && (end_w == 2 || end_w == 3) {
                     println!("DEBUG: Target Span Check: [{}-{}] Class {} Prob {}", start_w, end_w, j, prob);
                }
                
                if prob > 0.1 {
                     // Check span coordinates
                     let start_word = spans[i][0] as usize;
                     let end_word = spans[i][1] as usize;
                     
                     // Hybrid Decoding: Map WordID -> Tokens
                     // We need to reverse lookup: Which token mapped to start_word?
                     // Since mapping preserves order, we can find FIRST token with this word ID.
                     
                     let mut start_token = 0;
                     let mut end_token = 0;
                     
                     // Find first token for start_word
                     for (t_idx, &w_id) in mapping.iter().enumerate() {
                         if w_id == start_word as i64 {
                             start_token = t_idx;
                             break;
                         }
                     }

                     // Find LAST token for (end_word - 1)
                     // Since end is exclusive, we want the word before it.
                     let target_end_word = if end_word > 0 { end_word - 1 } else { 0 };
                     for (t_idx, &w_id) in mapping.iter().enumerate().rev() { // rev to find last
                         if w_id == target_end_word as i64 {
                             end_token = t_idx;
                             break;
                         }
                     }
                     
                     // text_encoding.get_offsets() gets char offsets for tokens.
                     // Ensure within bounds
                     let mut start_char = usize::MAX;
                     let mut end_char = 0;
                     
                     if let Some(offsets) = text_encoding.get_offsets().get(start_token) {
                          start_char = offsets.0;
                     }
                     if let Some(offsets) = text_encoding.get_offsets().get(end_token) {
                          end_char = offsets.1;
                     }
                     
                     if start_char < end_char {
                        let entity_text = text[start_char..end_char].to_string();
                        // Debug print
                        println!("DEBUG: Candidate Scan: [{}..{}] Text='{}' Label='{}' Score={}", 
                                  start_token, end_token, entity_text, labels[j], prob);

                        if prob > threshold {
                            entities.push(Entity {
                                start: start_char,
                                end: end_char,
                                text: entity_text,
                                label: labels[j].to_string(),
                                score: prob,
                            });
                        }
                     }
                }
            }
        }
        
        // NMS (Non-Maximum Suppression)
        // Sort by score
        entities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        println!("DEBUG: Raw entities count: {}", entities.len());
        for e in entities.iter().take(10) {
             println!("DEBUG: Candidate: {:?} - Score: {}", e.text, e.score);
        }
        
        let mut kept_entities: Vec<Entity> = Vec::new();
        // Simple NMS: if overlap, pick higher score
        for e in entities {
            let mut overlap = false;
            for kept in &kept_entities {
                if e.start < kept.end && e.end > kept.start {
                    overlap = true; // Strict overlap
                    break;
                }
            }
            if !overlap {
                kept_entities.push(e);
            }
        }

        Ok(kept_entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gliner_v2_inference() {
        // Path relative to cargo manifest
        let model_path = "models/gliner_multi_v2.1";
        if !std::path::Path::new(model_path).exists() {
            println!("⚠️ Skipping test: Model not found at {}", model_path);
            return;
        }

        println!("🚀 Loading GLiNER v2 from: {}", model_path);
        match GlinerModel::new(model_path) {
            Ok(mut model) => {
                let text = "Cristiano Ronaldo plays for Al-Nassr in Saudi Arabia.";
                let labels = vec!["person", "organization", "location"];
                
                let start = std::time::Instant::now();
                match model.predict_entities(text, &labels, 0.3) {
                    Ok(entities) => {
                         println!("✅ Inference took: {:?}", start.elapsed());
                         println!("📍 Entities found: {:?}", entities);
                         
                         assert!(!entities.is_empty(), "Should detect entities");
                         
                         // Check for specific entity
                         let has_person = entities.iter().any(|e| e.label == "person" && e.text.contains("Ronaldo"));
                         assert!(has_person, "Should detect Cristiano Ronaldo as person");
                    },
                    Err(e) => panic!("Inference failed: {}", e),
                }
            },
            Err(e) => panic!("Failed to load model: {}", e),
        }
    }
}
