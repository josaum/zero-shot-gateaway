use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A registry for document schemas and their extraction prompts.
#[derive(Debug, Clone)]
pub struct SchemaRegistry {
    schemas: HashMap<String, DocumentSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSchema {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub fields: Vec<SchemaField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub field_type: String,
    pub description: String,
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            schemas: HashMap::new(),
        };
        registry.register_defaults();
        registry
    }

    fn register_defaults(&mut self) {
        // 1. Invoice
        self.register(DocumentSchema {
            name: "invoice".to_string(),
            description: "Commercial invoice or receipt".to_string(),
            system_prompt: r#"Extract the following fields from the invoice text:
- invoice_number (string)
- date (ISO 8601 date string)
- total_amount (number)
- currency (string, e.g. BRL, USD)
- vendor_name (string)
- vendor_tax_id (string, e.g. CNPJ/CPF)
- items (array of objects with description, quantity, unit_price, total)

Return purely JSON."#.to_string(),
            fields: vec![
                SchemaField { name: "invoice_number".into(), field_type: "string".into(), description: "Invoice identifier".into() },
                SchemaField { name: "total_amount".into(), field_type: "number".into(), description: "Total value".into() },
            ],
        });

        // 2. Identity Document (RG/CNH - Brazil context)
        self.register(DocumentSchema {
            name: "identity".to_string(),
            description: "Personal identity document (RG, CNH, Passport)".to_string(),
            system_prompt: r#"Extract the following fields from the identity document:
- full_name (string)
- document_number (string, RG or CPF)
- birth_date (ISO 8601 date string)
- mother_name (string)
- father_name (string)
- issuing_authority (string)

Return purely JSON."#.to_string(),
            fields: vec![
                SchemaField { name: "full_name".into(), field_type: "string".into(), description: "Full name".into() },
                SchemaField { name: "document_number".into(), field_type: "string".into(), description: "RG or CPF".into() },
            ],
        });

         // 3. Receipt (Generic)
        self.register(DocumentSchema {
            name: "receipt".to_string(),
            description: "Generic payment receipt".to_string(),
            system_prompt: r#"Extract the following fields from the receipt:
- merchant_name (string)
- date (ISO 8601 date string)
- total (number)
- tax_id (string, optional)

Return purely JSON."#.to_string(),
            fields: vec![
                SchemaField { name: "merchant_name".into(), field_type: "string".into(), description: "Merchant".into() },
                SchemaField { name: "total".into(), field_type: "number".into(), description: "Total paid".into() },
            ],
        });
    }

    pub fn register(&mut self, schema: DocumentSchema) {
        self.schemas.insert(schema.name.clone(), schema);
    }

    #[allow(dead_code)]
    pub fn get(&self, name: &str) -> Option<&DocumentSchema> {
        self.schemas.get(name)
    }

    /// Improve the LLM system prompt based on the requested schema name.
    /// Returns the schema-specific prompt if found, otherwise a generic fallback.
    pub fn get_extraction_prompt(&self, schema_name: &str) -> String {
        if let Some(schema) = self.schemas.get(schema_name) {
            schema.system_prompt.clone()
        } else {
            // Fallback generic extraction
            r#"Analyze the provided text and extract semantic structures into a JSON object.
Infer key-value pairs that represent the core entities in the document.
Return purely JSON."#.to_string()
        }
    }
}
