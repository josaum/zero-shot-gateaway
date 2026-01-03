use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::copy;
use tracing::info;

pub struct ModelManager;

impl ModelManager {
    /// Ensures that a model exists at the destination path.
    /// If not, it attempts to download it from the source URL.
    pub fn ensure_model(name: &str, url: &str, dest: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if dest.exists() {
            info!("✅ Model '{}' found at {:?}", name, dest);
            return Ok(dest.to_path_buf());
        }

        info!("⬇️  Downloading model '{}' from {}...", name, url);
        
        // Ensure parent directory exists
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut response = reqwest::blocking::get(url)?;
        if !response.status().is_success() {
             return Err(format!("Failed to download model '{}': Status {}", name, response.status()).into());
        }

        let mut dest_file = File::create(dest)?;
        match copy(&mut response, &mut dest_file) {
            Ok(bytes) => info!("📦 Wrote {} bytes to {:?}", bytes, dest),
            Err(e) => {
                let _ = fs::remove_file(dest); // Cleanup partial file
                return Err(format!("Failed to write model '{}': {}", name, e).into());
            }
        }

        info!("🎉 Model '{}' downloaded successfully to {:?}", name, dest);
        Ok(dest.to_path_buf())
    }
}
