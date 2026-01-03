//! OCR Pipeline Abstraction
//!
//! Provides a unified interface for different OCR/document understanding pipelines:
//! - `Legacy`: Custom PP-StructureV3 implementation
//! - `OarStructure`: OAR-OCR OARStructureBuilder
//! - `OarVL`: PaddleOCR-VL Vision-Language model



/// OCR Pipeline Configuration
#[derive(Debug, Clone, Default)]
pub enum PipelineType {
    #[default]
    Legacy,
    #[cfg(feature = "oar")]
    OarStructure,
    #[cfg(feature = "vl")]
    OarVL,
}

impl std::str::FromStr for PipelineType {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            #[cfg(feature = "oar")]
            "oar-structure" | "oar" => Ok(Self::OarStructure),
            #[cfg(feature = "vl")]
            "oar-vl" | "vl" => Ok(Self::OarVL),
            _ => Err(format!("Unknown pipeline: {}. Valid options: legacy, oar-structure, oar-vl", s)),
        }
    }
}

/// Result of document processing
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub markdown: String,
    pub _html_tables: Vec<String>,
    pub _latex_formulas: Vec<String>,
}

impl DocumentResult {
    pub fn new(markdown: String) -> Self {
        Self {
            markdown,
            _html_tables: Vec::new(),
            _latex_formulas: Vec::new(),
        }
    }
}

/// Unified OCR Pipeline trait
pub trait OcrPipelineTrait: Send + Sync {
    /// Process a file path (preferred for disk-based models)
    fn process_file(&mut self, path: &str) -> Result<DocumentResult, Box<dyn std::error::Error>>;
    
    fn name(&self) -> &'static str;
}

/// Legacy pipeline wrapper (delegates to existing Collider methods)
pub struct LegacyPipeline;

impl OcrPipelineTrait for LegacyPipeline {

    fn process_file(&mut self, _path: &str) -> Result<DocumentResult, Box<dyn std::error::Error>> {
        Ok(DocumentResult::new("[Legacy Logic Loop]".to_string()))
    }
    
    fn name(&self) -> &'static str {
        "legacy"
    }
}

#[cfg(feature = "oar")]
pub mod oar_structure {
    use super::*;
    use oar_ocr::prelude::*;
    
    /// OAR-OCR Structure Pipeline using OARStructureBuilder
    pub struct OarStructurePipeline {
        structure: OARStructure,
    }
    
    impl OarStructurePipeline {
        /// Create a new OAR Structure Pipeline
        /// 
        /// # Arguments
        /// * `layout_model` - Path to layout detection ONNX model (e.g., pp-doclayout_plus-l.onnx)
        /// * `table_cls_model` - Optional path to table classification model
        /// * `table_cell_model` - Optional path to table cell detection model (RT-DETR)
        /// * `table_structure_model` - Path to table structure recognition model (SLANet/SLANeXt)
        /// * `table_structure_dict` - Path to table structure dictionary
        /// * `det_model` - Path to text detection ONNX model
        /// * `rec_model` - Path to text recognition ONNX model
        /// * `rec_dict` - Path to recognition character dictionary
        pub fn new(
            layout_model: &str,
            table_cls_model: Option<&str>,
            table_cell_model: Option<&str>,
            table_structure_model: &str,
            table_structure_dict: &str,
            det_model: &str,
            rec_model: &str,
            rec_dict: &str,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            // Determine layout model name preset
            let layout_name = if layout_model.contains("plus") {
                "pp-doclayout_plus-l"
            } else if layout_model.contains("picodet") {
                "picodet_layout_1x"
            } else {
                "pp-doclayout-l"
            };

            let mut builder = OARStructureBuilder::new(layout_model)
                .layout_model_name(layout_name)
                .with_wired_table_structure(table_structure_model)
                .wired_table_structure_model_name(if table_structure_model.contains("slanext") { "SLANeXt_wired" } else { "SLANet" })
                .table_structure_dict_path(table_structure_dict)
                .with_ocr(det_model, rec_model, rec_dict)
                .text_detection_model_name(if det_model.contains("server") { "PP-OCRv5_server_det" } else { "PP-OCRv5_mobile_det" })
                .text_recognition_model_name("PP-OCRv5_mobile_rec"); // Latin/Common Mobile Rec
            
            if let Some(cls) = table_cls_model {
                builder = builder.with_table_classification(cls);
            }
            
            if let Some(cell) = table_cell_model {
                builder = builder.with_wired_table_cell_detection(cell)
                    .wired_table_cell_model_name("RT-DETR-L_wired_table_cell_det");
            }

            // Also add a wireless fallback if available in models dir
            if std::path::Path::new("models/slanet_plus.onnx").exists() {
                builder = builder.with_wireless_table_structure("models/slanet_plus.onnx")
                    .wireless_table_structure_model_name("SLANet_plus");
            }
            
            let structure = builder.build()?;
            Ok(Self { structure })
        }
        
        /// Process an image file path and return structured results
        pub fn process_file_impl(&self, path: &str) -> Result<DocumentResult, Box<dyn std::error::Error>> {
            println!("🔍 OAR Debug: Attempting to process file: {}", path);
            let result = self.structure.predict(path)?;
            
            // Debug Log:
            println!("🔍 OAR Debug: Detected {} layout elements in {}", result.layout_elements.len(), path);
            for (i, elem) in result.layout_elements.iter().enumerate() {
                 println!("  [{}] Type: {:?}, Box: {:?}, Text Len: {}", i, elem.element_type, elem.bbox, elem.text.as_ref().map(|s| s.len()).unwrap_or(0));
            }

            
            // Use built-in to_markdown() for layout + tables + formulas
            let markdown = result.to_markdown();
            
            // Extract HTML tables
            let html_tables: Vec<String> = result.tables
                .iter()
                .filter_map(|t| t.html_structure.clone())
                .collect();
            
            // Extract formulas (using element text for formulas)
            let latex_formulas: Vec<String> = result.layout_elements
                .iter()
                .filter(|e| matches!(e.element_type, 
                    oar_ocr::domain::structure::LayoutElementType::Formula |
                    oar_ocr::domain::structure::LayoutElementType::FormulaNumber))
                .filter_map(|e| e.text.clone())
                .collect();
            
            Ok(DocumentResult {
                markdown,
                _html_tables: html_tables,
                _latex_formulas: latex_formulas,
            })
        }
    }
    
    impl OcrPipelineTrait for OarStructurePipeline {

        fn process_file(&mut self, path: &str) -> Result<DocumentResult, Box<dyn std::error::Error>> {
            self.process_file_impl(path)
        }
        
        fn name(&self) -> &'static str {
            "oar-structure"
        }
    }
}

#[cfg(feature = "vl")]
pub mod oar_vl {
    use super::*;
    use oar_ocr::vl::{PaddleOcrVl, PaddleOcrVlTask};
    
    /// PaddleOCR-VL Vision-Language Pipeline
    /// 
    /// A ~2B parameter Vision-Language Model for document understanding:
    /// - OCR: Text recognition
    /// - Table: HTML table reconstruction
    /// - Formula: LaTeX formula recognition
    /// - Chart: Chart/diagram understanding
    /// 
    /// Requires GPU for reasonable performance (~2GB VRAM)
    pub struct OarVLPipeline {
        model: PaddleOcrVl,
        max_new_tokens: usize,
    }
    
    impl OarVLPipeline {
        /// Create a new VL Pipeline from model directory
        /// 
        /// # Arguments
        /// * `model_dir` - Path to PaddleOCR-VL model directory (from HuggingFace)
        /// * `use_cuda` - Whether to use CUDA (recommended for performance)
        pub fn new(model_dir: &str, use_cuda: bool) -> Result<Self, Box<dyn std::error::Error>> {
            // Access Device via candle_core (re-exported by oar-ocr when vl feature is enabled)
            let device = if use_cuda {
                #[cfg(feature = "cuda")]
                {
                    candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    candle_core::Device::Cpu
                }
            } else {
                candle_core::Device::Cpu
            };
            
            let model = PaddleOcrVl::from_dir(model_dir, device)?;
            
            Ok(Self {
                model,
                max_new_tokens: 4096, // Reasonable default for documents
            })
        }
        
        /// Create VL Pipeline with CPU (slower, but works without GPU)
        #[allow(dead_code)]
        pub fn new_cpu(model_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
            Self::new(model_dir, false)
        }
        
        /// Process an image for OCR (text recognition)
        #[allow(dead_code)]
        pub fn ocr(&self, img: image::RgbImage) -> Result<String, Box<dyn std::error::Error>> {
            Ok(self.model.generate(img, PaddleOcrVlTask::Ocr, self.max_new_tokens)?)
        }
        
        /// Process an image for table recognition (returns HTML)
        #[allow(dead_code)]
        pub fn table(&self, img: image::RgbImage) -> Result<String, Box<dyn std::error::Error>> {
            Ok(self.model.generate(img, PaddleOcrVlTask::Table, self.max_new_tokens)?)
        }
        
        /// Process an image for formula recognition (returns LaTeX)
        #[allow(dead_code)]
        pub fn formula(&self, img: image::RgbImage) -> Result<String, Box<dyn std::error::Error>> {
            Ok(self.model.generate(img, PaddleOcrVlTask::Formula, self.max_new_tokens)?)
        }
        
        /// Process an image for chart recognition
        #[allow(dead_code)]
        pub fn chart(&self, img: image::RgbImage) -> Result<String, Box<dyn std::error::Error>> {
            Ok(self.model.generate(img, PaddleOcrVlTask::Chart, self.max_new_tokens)?)
        }
        
        /// Process a document image with automatic content type detection
        pub fn process_document_impl(&mut self, path: &str) -> Result<DocumentResult, Box<dyn std::error::Error>> {
            let img = image::open(path)?.to_rgb8();
            
            // For VL, we default to full OCR which recovers everything (inc. tables/formulas in textual form)
            let markdown = self.model.generate(img.clone(), PaddleOcrVlTask::Ocr, self.max_new_tokens)?;
            
            // Additionally extract specifically for tables/formulas if requested or just use the unified output
            let mut html_tables = Vec::new();
            if let Ok(table) = self.model.generate(img.clone(), PaddleOcrVlTask::Table, self.max_new_tokens) {
                html_tables.push(table);
            }
            
            let mut latex_formulas = Vec::new();
            if let Ok(formula) = self.model.generate(img, PaddleOcrVlTask::Formula, self.max_new_tokens) {
                 latex_formulas.push(formula);
            }

            Ok(DocumentResult {
                markdown,
                _html_tables: html_tables,
                _latex_formulas: latex_formulas,
            })
        }
    }
    
    impl OcrPipelineTrait for OarVLPipeline {

        fn process_file(&mut self, path: &str) -> Result<DocumentResult, Box<dyn std::error::Error>> {
            self.process_document_impl(path)
        }
        
        fn name(&self) -> &'static str {
            "oar-vl"
        }
    }
}
