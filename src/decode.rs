//! Decoder v3 — 1:1 port of the reference implementation `decode.py`. Order fixed by `apply`:
//! particle strip → RRN/FRN 7th-digit relabel → fragment drop. Verified against
//! `tests/fixtures/decoder_vectors.json` (shared with the Python reference).
use crate::types::{EntityType, Span};

pub const DECODER_VERSION: &str = "v3_particle_strip_rrn_frn_7th_fragment_drop";

/// Types whose values end in a digit — a trailing particle is never part of the value.
const NUMERIC_TAIL_TYPES: [EntityType; 8] = [
    EntityType::Brn, EntityType::Rrn, EntityType::Frn, EntityType::DriverLicense,
    EntityType::Phone, EntityType::Card, EntityType::Account, EntityType::Passport,
];
/// Single source of truth for particles (strip and fragment-drop share it). Longest first.
pub const PARTICLES: [&str; 18] = [
    "으로", "에게", "부터", "까지", "를", "을", "가", "이", "는", "은", "로", "와", "과", "의", "에", "도", "만", "께",
];

/// Python `str.isdigit()` approximation: Unicode decimal digits plus other numeric digit chars (No).
fn py_isdigit(c: char) -> bool { c.is_numeric() && !is_letter_number(c) }
fn is_letter_number(c: char) -> bool { matches!(c, 'Ⅰ'..='ↈ') }   // Nl (roman numerals etc.) are not isdigit
fn py_isalnum(c: char) -> bool { c.is_alphanumeric() }

fn trailing_particle(seg: &[char]) -> Option<usize> {
    // longest particle first so "으로" is not cut as "로", "에게" not as "에"
    let mut cands: Vec<&str> = PARTICLES.to_vec();
    cands.sort_by_key(|p| std::cmp::Reverse(p.chars().count()));
    for p in cands {
        let pc: Vec<char> = p.chars().collect();
        if seg.len() >= pc.len() && seg[seg.len() - pc.len()..] == pc[..] {
            return Some(pc.len());
        }
    }
    None
}

pub fn strip_trailing_particles(chars: &[char], spans: &[Span]) -> Vec<Span> {
    spans.iter().map(|s| {
        if NUMERIC_TAIL_TYPES.contains(&s.entity) {
            let seg = &chars[s.start..s.end];
            if let Some(n) = trailing_particle(seg) {
                let end = s.end - n;
                if end > s.start && py_isdigit(chars[end - 1]) {
                    return Span { end, ..s.clone() };
                }
            }
        }
        s.clone()
    }).collect()
}

pub fn relabel_rrn_frn_by_seventh_digit(chars: &[char], spans: &[Span]) -> Vec<Span> {
    spans.iter().map(|s| {
        if matches!(s.entity, EntityType::Rrn | EntityType::Frn) {
            let seg = &chars[s.start..s.end];
            if seg.len() == 13 && seg.iter().all(|c| c.is_ascii_digit()) {
                let want = match seg[6] { '1'..='4' => Some(EntityType::Rrn), '5'..='8' => Some(EntityType::Frn), _ => None };
                if let Some(w) = want {
                    if w != s.entity { return Span { entity: w, ..s.clone() }; }
                }
            }
        }
        s.clone()
    }).collect()
}

pub fn drop_fragments(chars: &[char], spans: &[Span]) -> Vec<Span> {
    let kept: Vec<&Span> = spans.iter().filter(|s| chars[s.start..s.end].iter().any(|c| py_isalnum(*c))).collect();
    let mut out = Vec::with_capacity(kept.len());
    for (i, s) in kept.iter().enumerate() {
        if s.entity != EntityType::Name {
            let core: String = chars[s.start..s.end].iter().filter(|c| !c.is_whitespace()).collect();
            if PARTICLES.contains(&core.as_str())
                && kept.iter().enumerate().any(|(j, o)| j != i && o.entity == s.entity && s.start >= o.end && s.start - o.end <= 1)
            {
                continue;
            }
        }
        out.push((*s).clone());
    }
    out
}

/// Full backend-contract decoder.
pub fn apply(chars: &[char], spans: &[Span]) -> Vec<Span> {
    drop_fragments(chars, &relabel_rrn_frn_by_seventh_digit(chars, &strip_trailing_particles(chars, spans)))
}
