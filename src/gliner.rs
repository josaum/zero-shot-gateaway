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

        // 3. Prepare Masks & Spans using SentencePiece subword grouping
        // GLiNER groups tokens into "words" based on the ▁ (U+2581) prefix

        let seq_len = input_ids.len();
        let attention_mask_array = Array2::<i64>::ones((batch_size, seq_len));
        let mut words_mask_array = Array2::<i64>::zeros((batch_size, seq_len));

        let num_text_tokens = text_ids.len();
        let tokens = text_encoding.get_tokens();
        let offsets = text_encoding.get_offsets();

        // Debug tokenizer output
        println!("DEBUG: Tokens and offsets:");
        for (i, t) in tokens.iter().enumerate() {
            println!("  Token {}: '{}' offsets={:?}", i, t, offsets.get(i));
        }

        // Map tokens to word IDs using SentencePiece ▁ prefix
        // A new word starts when:
        // 1. Token begins with ▁ (and has content after it)
        // 2. Previous token was a standalone ▁ (space separator)
        // 3. It's the first content token
        let mut mapping: Vec<i64> = vec![0; seq_len];
        let mut current_word_id: i64 = 0;

        // Track word char spans for entity extraction later
        let mut word_char_spans: Vec<(usize, usize)> = Vec::new();
        let mut current_word_start: Option<usize> = None;
        let mut after_standalone_space = false;

        for token_idx in 0..num_text_tokens {
            let token = &tokens[token_idx];

            // Special tokens get 0
            if token == "[CLS]" || token == "[SEP]" || token == "[PAD]" {
                mapping[token_idx] = 0;
                // End current word if any
                if let Some(start) = current_word_start {
                    if token_idx > 0 {
                        if let Some(&(_, prev_end)) = offsets.get(token_idx - 1) {
                            word_char_spans.push((start, prev_end));
                        }
                    }
                    current_word_start = None;
                }
                after_standalone_space = false;
                continue;
            }

            // Check token type
            let starts_with_space_marker = token.starts_with('▁');
            let is_standalone_space = token == "▁";
            let has_content_after_space = starts_with_space_marker && token.len() > "▁".len();

            if is_standalone_space {
                // Standalone space token - marks word boundary
                if let Some(start) = current_word_start {
                    if token_idx > 0 {
                        if let Some(&(_, prev_end)) = offsets.get(token_idx - 1) {
                            word_char_spans.push((start, prev_end));
                        }
                    }
                    current_word_start = None;
                }
                mapping[token_idx] = 0;
                after_standalone_space = true;
                continue;
            }

            // Determine if this starts a new word
            let starts_new_word = has_content_after_space ||
                                  after_standalone_space ||
                                  current_word_id == 0;

            if starts_new_word {
                // End previous word if any
                if let Some(start) = current_word_start {
                    if token_idx > 0 {
                        if let Some(&(_, prev_end)) = offsets.get(token_idx - 1) {
                            word_char_spans.push((start, prev_end));
                        }
                    }
                }
                // Start new word
                current_word_id += 1;
                if let Some(&(tok_start, _)) = offsets.get(token_idx) {
                    current_word_start = Some(tok_start);
                }
            }

            // Map token to current word
            mapping[token_idx] = current_word_id;
            after_standalone_space = false;
        }

        // End last word if any
        if let Some(start) = current_word_start {
            if let Some(&(_, last_end)) = offsets.get(num_text_tokens.saturating_sub(1)) {
                word_char_spans.push((start, last_end));
            }
        }

        let num_words = current_word_id as usize;

        // Label tokens and padding get 0
        for i in num_text_tokens..seq_len {
            mapping[i] = 0;
        }

        // Apply to array
        for i in 0..seq_len {
            words_mask_array[[0, i]] = mapping[i];
        }

        println!("DEBUG: Num words (from SP grouping): {}", num_words);
        println!("DEBUG: Word char spans: {:?}", word_char_spans);
        println!("DEBUG: Full mapping (text portion): {:?}", &mapping[..num_text_tokens.min(mapping.len())]);

        // 4. Generate Spans over content token positions
        // The model expects spans over the positions of tokens with word_id > 0
        // First, find the indices of content tokens
        let content_token_indices: Vec<usize> = (0..num_text_tokens)
            .filter(|&i| mapping[i] > 0)
            .collect();
        let num_content_tokens = content_token_indices.len();

        println!("DEBUG: Content token indices: {:?}", content_token_indices);
        println!("DEBUG: num_content_tokens = {}", num_content_tokens);

        let max_width = self.config.max_width;
        let mut spans: Vec<[i64; 2]> = Vec::new();

        // Generate spans over content token positions (0-indexed)
        for start in 0..num_content_tokens {
            for width in 1..=max_width {
                let end = start + width - 1; // inclusive end
                if end < num_content_tokens {
                    spans.push([start as i64, end as i64]);
                } else {
                    // Padding span
                    spans.push([0, 0]);
                }
            }
        }

        println!("DEBUG: Generated {} spans for {} content tokens", spans.len(), num_content_tokens);
    
        // Construct Span Arrays
        let num_spans = spans.len();
        let mut span_idx_array = Array3::<i64>::zeros((batch_size, num_spans, 2));
        let mut span_mask_array = Array2::<bool>::from_elem((batch_size, num_spans), false);

        for (i, span) in spans.iter().enumerate() {
            span_idx_array[[0, i, 0]] = span[0];
            span_idx_array[[0, i, 1]] = span[1];

            // Mark valid spans (not padding spans)
            // Valid if end < num_content_tokens and it's not a pure padding span
            let is_valid_span = (span[1] as usize) < num_content_tokens;
            // Also check it's not [0,0] padding when start > 0 was expected
            let is_meaningful = i < (num_content_tokens * max_width) &&
                               (span[0] != 0 || span[1] != 0 || i == 0);
            span_mask_array[[0, i]] = is_valid_span || (span[0] == 0 && span[1] == 0 && i == 0);
        }

        // The model internally computes word embeddings by counting tokens with word_id > 0
        // So text_lengths should match the number of content tokens, not unique words
        let content_token_count = mapping[..num_text_tokens]
            .iter()
            .filter(|&&x| x > 0)
            .count();
        let text_length_value = content_token_count as i64;
        println!("DEBUG: text_lengths = {} (content tokens)", text_length_value);
        println!("DEBUG: num_words = {} (unique word IDs)", num_words);

        let mut text_lengths_array = Array2::<i64>::zeros((batch_size, 1));
        text_lengths_array[[0, 0]] = text_length_value; 
        
        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)?;

        // Debug inputs before run
        println!("DEBUG: Span count: {}", num_spans);
        println!("DEBUG: First 5 spans: {:?}", &spans[..5.min(spans.len())]);

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

        let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;

        // 6. Decode Output
        // ONNX Output Shape: [Batch, NumWords, MaxWidth, NumClasses]
        println!("DEBUG: Output shape: {:?}", shape);
        println!("DEBUG: Output data len: {}", data.len());

        let num_words_out = shape[1] as usize;
        let max_width_out = shape[2] as usize;
        let num_classes_out = shape[3] as usize;

        println!("DEBUG: num_words_out={}, max_width_out={}, num_classes_out={}",
                 num_words_out, max_width_out, num_classes_out);

        let mut entities = Vec::new();

        // Iterate over the output tensor directly
        // Layout: [batch, word_position, span_width, class]
        for word_pos in 0..num_words_out.min(num_words) {
            for width_idx in 0..max_width_out {
                let span_end = word_pos + width_idx; // inclusive end
                if span_end >= num_words {
                    continue; // Out of bounds
                }

                for class_idx in 0..labels.len().min(num_classes_out) {
                    // Flat index into the data tensor
                    let flat_idx = word_pos * (max_width_out * num_classes_out)
                                 + width_idx * num_classes_out
                                 + class_idx;

                    if flat_idx >= data.len() {
                        continue;
                    }

                    let score = data[flat_idx];
                    let prob = 1.0 / (1.0 + (-score).exp()); // sigmoid

                    // Debug: Check interesting spans
                    if word_pos <= 1 && span_end <= 2 && prob > 0.05 {
                        println!("DEBUG: Span [{}-{}] class={} prob={:.4}",
                                 word_pos, span_end, labels[class_idx], prob);
                    }

                    if prob > threshold {
                        // Map word positions directly to character offsets using word_char_spans
                        let start_char = word_char_spans[word_pos].0;
                        let end_char = word_char_spans[span_end].1;

                        let entity_text = text[start_char..end_char].to_string();

                        println!("DEBUG: Found entity: '{}' ({}) prob={:.4}",
                                 entity_text, labels[class_idx], prob);

                        entities.push(Entity {
                            start: start_char,
                            end: end_char,
                            text: entity_text,
                            label: labels[class_idx].to_string(),
                            score: prob,
                        });
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
