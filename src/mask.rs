//! Span geometry and rendering. Overlaps: higher score wins, ties → longer span, then earlier start; the
//! loser keeps whatever part of it lies outside already-claimed characters, so an overlap never leaves
//! part of a lower-scoring span in clear text. Replacement text is a per-type policy (`MaskTokens`),
//! default `[TYPE]`.
use crate::types::{EntityType, Span};

/// Replacement text per entity type, indexed by position in `EntityType::ALL`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskTokens([String; EntityType::ALL.len()]);

impl Default for MaskTokens {
    /// `[PHONE]`, `[NAME]`, `[DRIVER_LICENSE]`, `[NUM]`, ... — the type name in square brackets.
    fn default() -> Self { MaskTokens(EntityType::ALL.map(|t| format!("[{}]", t.as_str()))) }
}

impl MaskTokens {
    pub fn get(&self, t: EntityType) -> &str { &self.0[Self::idx(t)] }
    /// Override one type's replacement (an empty string deletes the span's text).
    pub fn set(&mut self, t: EntityType, token: impl Into<String>) { self.0[Self::idx(t)] = token.into(); }
    fn idx(t: EntityType) -> usize { EntityType::ALL.iter().position(|x| *x == t).expect("ALL lists every variant") }
}

/// Remove `zones` from `s`, keeping the left/right remainders (template-variable protection, residual pieces).
pub fn clip(s: &Span, zones: &[(usize, usize)]) -> Vec<Span> {
    let mut pieces = vec![(s.start, s.end)];
    for &(zs, ze) in zones {
        let mut next = Vec::with_capacity(pieces.len() + 1);
        for (a, b) in pieces {
            if b <= zs || a >= ze { next.push((a, b)); continue; }
            if a < zs { next.push((a, zs)); }
            if b > ze { next.push((ze, b)); }
        }
        pieces = next;
    }
    pieces.into_iter().filter(|(a, b)| b > a).map(|(a, b)| Span::new(a, b, s.entity, s.score)).collect()
}

fn ranges(spans: &[Span]) -> Vec<(usize, usize)> { spans.iter().map(|s| (s.start, s.end)).collect() }

/// Winner order: score desc, longer first, earlier first, then type name. Each later span keeps only the
/// parts outside characters already claimed. Output is non-overlapping and sorted by start.
pub fn resolve_overlaps(spans: &[Span]) -> Vec<Span> {
    let mut order: Vec<&Span> = spans.iter().collect();
    order.sort_by(|a, b| b.score.total_cmp(&a.score)
        .then((b.end - b.start).cmp(&(a.end - a.start))).then(a.start.cmp(&b.start)).then(a.entity.cmp(&b.entity)));
    let mut kept: Vec<Span> = Vec::new();
    for s in order {
        let pieces = clip(s, &ranges(&kept));
        kept.extend(pieces);
    }
    kept.sort_by_key(|s| s.start);
    kept
}

/// Merge model spans with regex backstop spans (`backstop::find`) under template-variable protection:
/// both sides are clipped to the zones, backstop spans absorb each other (longer first, so EMAIL/URL
/// swallow the NUM run inside them), and a backstop span keeps only what the model did not already cover.
pub fn combine_backstop(ner: Vec<Span>, back: Vec<Span>, zones: &[(usize, usize)]) -> Vec<Span> {
    let ner: Vec<Span> = ner.iter().flat_map(|s| clip(s, zones)).collect();
    let mut back: Vec<Span> = back.iter().flat_map(|s| clip(s, zones)).collect();
    back.sort_by(|a, b| (b.end - b.start).cmp(&(a.end - a.start)).then(b.score.total_cmp(&a.score)).then(a.start.cmp(&b.start)));
    let mut back_kept: Vec<Span> = Vec::new();
    for s in &back {
        let pieces = clip(s, &ranges(&back_kept));
        back_kept.extend(pieces);
    }
    let ner_ranges = ranges(&ner);
    let mut out: Vec<Span> = ner;
    out.extend(back_kept.iter().flat_map(|s| clip(s, &ner_ranges)));
    out.sort_by_key(|s| (s.start, s.end));
    out
}

/// Adjacent pieces of the same type render as one token — a model span plus its backstop remainder
/// (`https://example.test/` + `a`) is one URL to the reader, not two.
pub fn render(chars: &[char], spans: &[Span], tokens: &MaskTokens) -> String {
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;
    let mut last: Option<(usize, EntityType)> = None;   // (end, type) of the previous rendered span
    for s in resolve_overlaps(spans) {
        if last == Some((s.start, s.entity)) { i = s.end; last = Some((s.end, s.entity)); continue; }
        out.extend(&chars[i..s.start]);
        out.push_str(tokens.get(s.entity));
        i = s.end;
        last = Some((s.end, s.entity));
    }
    out.extend(&chars[i..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntityType as E;

    fn c(s: &str) -> Vec<char> { s.chars().collect() }

    #[test]
    fn loser_keeps_residual_outside_winner() {
        // NAME 0..3 (.9) vs ADDRESS 0..10 (.8): without residuals chars 3..10 would stay in clear text.
        let spans = [Span::new(0, 3, E::Name, 0.9), Span::new(0, 10, E::Address, 0.8)];
        let kept = resolve_overlaps(&spans);
        assert_eq!(kept.iter().map(|s| (s.start, s.end, s.entity)).collect::<Vec<_>>(),
                   [(0, 3, E::Name), (3, 10, E::Address)]);
        assert_eq!(render(&c("홍길동 서울시 강남구"), &spans, &MaskTokens::default()), "[NAME][ADDRESS]구");
    }

    #[test]
    fn adjacent_same_type_pieces_render_as_one_token() {
        let chars = c("링크 https://example.test/a 끝");
        let spans = [Span::new(3, 24, E::Url, 0.9), Span::new(24, 25, E::Url, 0.6)];   // model + backstop remainder
        assert_eq!(render(&chars, &spans, &MaskTokens::default()), "링크 [URL] 끝");
        let mixed = [Span::new(3, 24, E::Url, 0.9), Span::new(24, 25, E::Num, 0.5)];   // different type stays separate
        assert_eq!(render(&chars, &mixed, &MaskTokens::default()), "링크 [URL][NUM] 끝");
    }

    #[test]
    fn clip_removes_zones_and_keeps_both_sides() {
        let s = Span::new(0, 10, E::Phone, 0.9);
        let out = clip(&s, &[(3, 5)]);
        assert_eq!(out.iter().map(|x| (x.start, x.end)).collect::<Vec<_>>(), [(0, 3), (5, 10)]);
        assert!(clip(&s, &[(0, 10)]).is_empty());
    }

    #[test]
    fn combine_backstop_absorbs_num_into_email_and_yields_to_model() {
        // "a1234567@b.com" → backstop EMAIL 0..14 and NUM 1..8 (digit run inside the email) → EMAIL absorbs NUM.
        let back = vec![Span::new(0, 14, E::Email, 0.6), Span::new(1, 8, E::Num, 0.5)];
        let out = combine_backstop(vec![], back.clone(), &[]);
        assert_eq!(out.iter().map(|s| (s.start, s.end, s.entity)).collect::<Vec<_>>(), [(0, 14, E::Email)]);
        // model already covers 0..9 → backstop keeps only the residual 9..14
        let out = combine_backstop(vec![Span::new(0, 9, E::Email, 0.95)], back, &[]);
        assert_eq!(out.iter().map(|s| (s.start, s.end, s.entity)).collect::<Vec<_>>(),
                   [(0, 9, E::Email), (9, 14, E::Email)]);
    }

    #[test]
    fn template_variables_are_never_masked() {
        // zone 3..9 = "#{이름}" inside a model span 0..12
        let out = combine_backstop(vec![Span::new(0, 12, E::Name, 0.9)], vec![Span::new(2, 11, E::Num, 0.5)], &[(3, 9)]);
        assert!(out.iter().all(|s| s.end <= 3 || s.start >= 9), "{out:?}");
    }

    #[test]
    fn mask_tokens_default_and_override() {
        let mut tokens = MaskTokens::default();
        assert_eq!(tokens.get(E::DriverLicense), "[DRIVER_LICENSE]");
        assert_eq!(tokens.get(E::Num), "[NUM]");
        tokens.set(E::Phone, "***");
        tokens.set(E::Name, "");
        let spans = vec![Span::new(0, 3, E::Name, 0.9), Span::new(4, 15, E::Phone, 0.9)];
        assert_eq!(render(&c("홍길동 01012345678"), &spans, &tokens), " ***");
    }
}
