//! Span → masked text. Overlaps: higher score wins, ties → longer span, then earlier start.
use crate::types::Span;

pub fn resolve_overlaps(spans: &[Span]) -> Vec<Span> {
    let mut order: Vec<&Span> = spans.iter().collect();
    order.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        .then((b.end - b.start).cmp(&(a.end - a.start))).then(a.start.cmp(&b.start)));
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
