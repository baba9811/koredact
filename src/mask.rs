//! Span → masked text. Overlaps: higher score wins, ties → longer span, then earlier start.
//! Replacement text is a per-type policy (`MaskTokens`), default `[TYPE]`.
use crate::types::{EntityType, Span};

/// Replacement text per entity type, indexed by position in `EntityType::ALL`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTokens([String; EntityType::ALL.len()]);

impl Default for MaskTokens {
    /// `[PHONE]`, `[NAME]`, `[DRIVER_LICENSE]`, ... — the type name in square brackets.
    fn default() -> Self { MaskTokens(EntityType::ALL.map(|t| format!("[{}]", t.as_str()))) }
}

impl MaskTokens {
    pub fn get(&self, t: EntityType) -> &str { &self.0[Self::idx(t)] }
    /// Override one type's replacement (an empty string deletes the span's text).
    pub fn set(&mut self, t: EntityType, token: impl Into<String>) { self.0[Self::idx(t)] = token.into(); }
    fn idx(t: EntityType) -> usize { EntityType::ALL.iter().position(|x| *x == t).expect("ALL lists every variant") }
}

pub fn resolve_overlaps(spans: &[Span]) -> Vec<Span> {
    let mut order: Vec<&Span> = spans.iter().collect();
    // total order: score desc (scores are finite — lib.rs rejects NaN logits), longer first, earlier first, then type
    order.sort_by(|a, b| b.score.total_cmp(&a.score)
        .then((b.end - b.start).cmp(&(a.end - a.start))).then(a.start.cmp(&b.start)).then(a.entity.cmp(&b.entity)));
    let mut kept: Vec<Span> = Vec::new();
    for s in order {
        if kept.iter().all(|k| s.end <= k.start || k.end <= s.start) { kept.push(s.clone()); }
    }
    kept.sort_by_key(|s| s.start);
    kept
}

pub fn render(chars: &[char], spans: &[Span], tokens: &MaskTokens) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;
    for s in resolve_overlaps(spans) {
        out.extend(&chars[i..s.start]);
        out.push_str(tokens.get(s.entity));
        i = s.end;
    }
    out.extend(&chars[i..]);
    out
}
