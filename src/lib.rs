//! koredact — Korean PII masking runtime: BERT token-classification (ONNX) + rule decoder.
//! Pipeline per text: char windows (400/100) → tokenize (char offsets) → ONNX logits → argmax →
//! simple grouping → window merge → decoder v3 → (optional regex backstop) → spans / masked text.
pub mod error;
pub mod types;

mod model;
mod rules;

// 디코더 버전·벡터는 배포 계약 — 경로를 내부 묶음(`rules`)에 묶지 않고 크레이트 최상위로 재노출
pub use rules::decode;

#[cfg(feature = "python")]
mod py;

use std::path::Path;

use serde::Deserialize;

pub use decode::DECODER_VERSION;
pub use error::Error;
pub use rules::mask::MaskTokens;
pub use types::{EntityType, Span};

#[derive(Deserialize)]
struct TokConfig { #[serde(default = "default_max_len")] model_max_length: usize }
fn default_max_len() -> usize { 512 }

pub struct Masker { tok: model::tokenize::Tok, labels: model::label::Labels, model: model::infer::Model, mask_tokens: MaskTokens }

impl Masker {
    /// Load a published bundle dir: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model.onnx`.
    pub fn from_dir(dir: &Path, threads: usize) -> Result<Masker, Error> {
        let need = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"];
        for n in need {
            if !dir.join(n).is_file() { return Err(Error::Bundle(format!("missing {n} in {}", dir.display()))); }
        }
        let tc: TokConfig = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)?;
        let labels = model::label::Labels::load(&dir.join("config.json"))?;
        let model = model::infer::Model::load(&dir.join("onnx/model.onnx"), labels.0.len(), threads)?;
        let tok = model::tokenize::Tok::load(&dir.join("tokenizer.json"), tc.model_max_length)?;
        Ok(Masker { tok, labels, model, mask_tokens: MaskTokens::default() })
    }

    pub fn mask_tokens(&self) -> &MaskTokens { &self.mask_tokens }
    pub fn set_mask_tokens(&mut self, tokens: MaskTokens) { self.mask_tokens = tokens; }

    /// Raw model spans (windows merged, decoder NOT applied).
    pub fn predict_raw(&mut self, text: &str) -> Result<Vec<Span>, Error> {
        let chars: Vec<char> = text.chars().collect();
        let mut spans = Vec::new();
        for off in model::window::starts(chars.len()) {
            let end = (off + model::window::MAX_CHARS).min(chars.len());
            let chunk: String = chars[off..end].iter().collect();
            let enc = self.tok.encode(&chunk)?;
            let rows = self.model.logits(enc.get_ids(), enc.get_type_ids(), enc.get_attention_mask())?;
            if rows.iter().any(|r| r.is_empty() || r.iter().any(|v| !v.is_finite())) {
                return Err(Error::Bundle("non-finite or empty logits row".into()));
            }
            let special = enc.get_special_tokens_mask();
            let offsets = enc.get_offsets();
            let toks: Vec<model::label::TokenPred> = rows.iter().enumerate()
                .filter(|(i, _)| special[*i] == 0)
                .map(|(i, row)| {
                    let (label, score) = argmax_softmax(row);
                    model::label::TokenPred { label, score, start: off + offsets[i].0, end: off + offsets[i].1 }
                }).collect();
            spans.extend(model::label::group_simple(&self.labels, &toks));
        }
        Ok(model::window::merge(spans))
    }

    /// Backend contract: raw spans + decoder v3 (no backstop, no type filter).
    pub fn predict(&mut self, text: &str) -> Result<Vec<Span>, Error> {
        self.predict_opts(text, None, false)
    }

    pub fn mask(&mut self, text: &str) -> Result<String, Error> {
        self.mask_opts(text, None, false)
    }

    /// `predict` restricted to `keep` types (see `predict_opts`).
    pub fn predict_types(&mut self, text: &str, keep: &[EntityType]) -> Result<Vec<Span>, Error> {
        self.predict_opts(text, Some(keep), false)
    }

    /// `mask` restricted to `keep` types; other types are left in clear text.
    pub fn mask_types(&mut self, text: &str, keep: &[EntityType]) -> Result<String, Error> {
        self.mask_opts(text, Some(keep), false)
    }

    /// Decoded spans with the two opt-ins. `backstop` merges the regex safety net (`rules::backstop::find`) under
    /// template-variable protection; `keep` then restricts types. The filter runs before overlap resolution,
    /// so a kept span is masked even where a dropped type scored higher on the same characters.
    pub fn predict_opts(&mut self, text: &str, keep: Option<&[EntityType]>, backstop: bool) -> Result<Vec<Span>, Error> {
        let chars: Vec<char> = text.chars().collect();
        let raw = self.predict_raw(text)?;
        let filter = |spans: Vec<Span>| match keep { Some(k) => keep_types(spans, k), None => spans };
        // filter each source before merging: a dropped model span must not clip (and then abandon) a kept span
        let spans = filter(rules::decode::apply(&chars, &raw));
        if !backstop { return Ok(spans); }
        let back = filter(rules::backstop::find(&chars));
        Ok(rules::mask::combine_backstop(spans, back, &rules::backstop::var_zones(&chars)))
    }

    pub fn mask_opts(&mut self, text: &str, keep: Option<&[EntityType]>, backstop: bool) -> Result<String, Error> {
        let chars: Vec<char> = text.chars().collect();
        let spans = self.predict_opts(text, keep, backstop)?;
        Ok(rules::mask::render(&chars, &spans, &self.mask_tokens))
    }
}

/// argmax + softmax probability of the argmax (transformers pipeline `scores` are softmaxed).
fn argmax_softmax(row: &[f32]) -> (usize, f32) {
    let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, v) in row.iter().enumerate() { if *v > bv { bv = *v; best = i; } }
    let denom: f32 = row.iter().map(|v| (v - bv).exp()).sum();
    (best, 1.0 / denom)
}

/// Drop spans whose type is not in `keep`. Runs on decoded spans, i.e. before `rules::mask::resolve_overlaps`,
/// so restricting to a type can never lose one of its spans to a higher-scoring span of another type.
fn keep_types(spans: Vec<Span>, keep: &[EntityType]) -> Vec<Span> {
    spans.into_iter().filter(|s| keep.contains(&s.entity)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_applies_to_both_sources_before_merge() {
        // model NAME 0..11 covers the digit run; with keep=[NUM] the NAME must not clip the NUM away
        let chars: Vec<char> = "01012345678".chars().collect();
        let ner = keep_types(vec![Span::new(0, 11, EntityType::Name, 0.9)], &[EntityType::Num]);
        let back = keep_types(rules::backstop::find(&chars), &[EntityType::Num]);
        let out = rules::mask::combine_backstop(ner, back, &[]);
        assert_eq!(out.iter().map(|s| (s.start, s.end, s.entity)).collect::<Vec<_>>(), [(0, 11, EntityType::Num)]);
    }

    #[test]
    fn kept_type_wins_over_dropped_higher_score_overlap() {
        let chars: Vec<char> = "홍길동 01012345678".chars().collect();
        // NAME (score .9) overlaps PHONE (score .5) on chars 2..6; unfiltered render keeps NAME only there.
        let spans = vec![Span::new(0, 6, EntityType::Name, 0.9), Span::new(2, 15, EntityType::Phone, 0.5)];
        let tokens = MaskTokens::default();
        // unfiltered: NAME wins 0..6, PHONE keeps its residual 6..15
        assert_eq!(rules::mask::render(&chars, &spans, &tokens), "[NAME][PHONE]");
        let only_phone = keep_types(spans.clone(), &[EntityType::Phone]);
        assert_eq!(only_phone.len(), 1);
        assert_eq!(rules::mask::render(&chars, &only_phone, &tokens), "홍길[PHONE]");
        assert!(keep_types(spans, &[]).is_empty());
    }
}
