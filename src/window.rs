//! 400/100 char windows + same-type overlap merge — mirrors `HfTokenClassifierBackend.predict` / `_merge_windows`.
use crate::types::Span;

pub const MAX_CHARS: usize = 400;
pub const OVERLAP: usize = 100;

/// Window start offsets: `range(0, max(len - overlap, 1), max_chars - overlap)`.
pub fn starts(n_chars: usize) -> Vec<usize> {
    let step = MAX_CHARS - OVERLAP;
    let stop = n_chars.saturating_sub(OVERLAP).max(1);
    (0..stop).step_by(step).collect()
}

/// Sort by (start, end); merge with the previous span when overlapping and same type (max end, max score).
pub fn merge(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by_key(|s| (s.start, s.end));
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for s in spans {
        if let Some(prev) = out.last() {
            if s.start < prev.end && s.entity == prev.entity {
                let prev = out.pop().unwrap();
                out.push(Span::new(prev.start, prev.end.max(s.end), s.entity, prev.score.max(s.score)));
                continue;
            }
        }
        out.push(s);
    }
    out
}
