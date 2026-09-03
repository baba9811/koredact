//! Span → masked text. Overlaps: higher score wins, ties → longer span, then earlier start.
use crate::types::Span;

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

pub fn render(chars: &[char], spans: &[Span]) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;
    for s in resolve_overlaps(spans) {
        out.extend(&chars[i..s.start]);
        out.push_str(s.entity.mask_token());
        i = s.end;
    }
    out.extend(&chars[i..]);
    out
}
