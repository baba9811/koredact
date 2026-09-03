//! Shared decoder vectors — byte-identical behaviour with the Python reference `decode.apply`.
use koredact::decode::{apply, DECODER_VERSION};
use koredact::types::{EntityType, Span};
use serde::Deserialize;

#[derive(Deserialize)]
struct RawSpan { start: usize, end: usize, entity: String }
#[derive(Deserialize)]
struct Case { name: String, text: String, spans: Vec<RawSpan>, expected: Vec<RawSpan> }
#[derive(Deserialize)]
struct Vectors { decoder_version: String, cases: Vec<Case> }

fn to_spans(v: &[RawSpan]) -> Vec<Span> {
    v.iter().map(|r| Span::new(r.start, r.end, EntityType::parse(&r.entity).expect("entity"), 1.0)).collect()
}

#[test]
fn decoder_matches_shared_vectors() {
    let v: Vectors = serde_json::from_str(include_str!("fixtures/decoder_vectors.json")).unwrap();
    assert_eq!(v.decoder_version, DECODER_VERSION);
    assert!(v.cases.len() >= 27);
    for c in &v.cases {
        let chars: Vec<char> = c.text.chars().collect();
        let got: Vec<(usize, usize, EntityType)> = apply(&chars, &to_spans(&c.spans)).iter().map(|s| (s.start, s.end, s.entity)).collect();
        let want: Vec<(usize, usize, EntityType)> = to_spans(&c.expected).iter().map(|s| (s.start, s.end, s.entity)).collect();
        assert_eq!(got, want, "case {}", c.name);
    }
}
