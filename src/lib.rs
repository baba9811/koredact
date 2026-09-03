//! koredact — Korean PII masking runtime: BERT token-classification (ONNX) + rule decoder.
//! Pipeline per text: char windows (400/100) → tokenize (char offsets) → ONNX logits → argmax →
//! simple grouping → window merge → decoder v3 → spans / masked text.
pub mod decode;
pub mod error;
pub mod infer;
pub mod label;
pub mod mask;
pub mod tokenize;
pub mod types;
pub mod window;
#[cfg(feature = "python")]
mod py;

use std::path::Path;

use serde::Deserialize;

pub use decode::DECODER_VERSION;
pub use error::Error;
pub use types::{EntityType, Span};

#[derive(Deserialize)]
struct TokConfig { #[serde(default = "default_max_len")] model_max_length: usize }
fn default_max_len() -> usize { 512 }

pub struct Masker { tok: tokenize::Tok, labels: label::Labels, model: infer::Model }

impl Masker {
    /// Load a published bundle dir: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model.onnx`.
    pub fn from_dir(dir: &Path, threads: usize) -> Result<Masker, Error> {
        let need = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"];
        for n in need {
            if !dir.join(n).is_file() { return Err(Error::Bundle(format!("missing {n} in {}", dir.display()))); }
        }
        let tc: TokConfig = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)?;
        let labels = label::Labels::load(&dir.join("config.json"))?;
        let model = infer::Model::load(&dir.join("onnx/model.onnx"), labels.0.len(), threads)?;
        let tok = tokenize::Tok::load(&dir.join("tokenizer.json"), tc.model_max_length)?;
        Ok(Masker { tok, labels, model })
    }

    /// Raw model spans (windows merged, decoder NOT applied).
    pub fn predict_raw(&mut self, text: &str) -> Result<Vec<Span>, Error> {
        let chars: Vec<char> = text.chars().collect();
        let mut spans = Vec::new();
        for off in window::starts(chars.len()) {
            let end = (off + window::MAX_CHARS).min(chars.len());
            let chunk: String = chars[off..end].iter().collect();
            let enc = self.tok.encode(&chunk)?;
            let rows = self.model.logits(enc.get_ids(), enc.get_type_ids(), enc.get_attention_mask())?;
            let special = enc.get_special_tokens_mask();
            let offsets = enc.get_offsets();
            let toks: Vec<label::TokenPred> = rows.iter().enumerate()
                .filter(|(i, _)| special[*i] == 0)
                .map(|(i, row)| {
                    let (label, score) = argmax_softmax(row);
                    label::TokenPred { label, score, start: off + offsets[i].0, end: off + offsets[i].1 }
                }).collect();
            spans.extend(label::group_simple(&self.labels, &toks));
        }
        Ok(window::merge(spans))
    }

    /// Backend contract: raw spans + decoder v3.
    pub fn predict(&mut self, text: &str) -> Result<Vec<Span>, Error> {
        let chars: Vec<char> = text.chars().collect();
        let raw = self.predict_raw(text)?;
        Ok(decode::apply(&chars, &raw))
    }

    pub fn mask(&mut self, text: &str) -> Result<String, Error> {
        let chars: Vec<char> = text.chars().collect();
        let spans = self.predict(text)?;
        Ok(mask::render(&chars, &spans))
    }
}

/// argmax + softmax probability of the argmax (transformers pipeline `scores` are softmaxed).
fn argmax_softmax(row: &[f32]) -> (usize, f32) {
    let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, v) in row.iter().enumerate() { if *v > bv { bv = *v; best = i; } }
    let denom: f32 = row.iter().map(|v| (v - bv).exp()).sum();
    (best, 1.0 / denom)
}
