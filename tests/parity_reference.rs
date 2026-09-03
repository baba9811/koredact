//! Parity with the Python reference backend (the reference implementation `HfTokenClassifierBackend` + `decode.apply`).
//! Needs a real bundle and a reference dump → runs only when both env vars are set:
//!   KOREDACT_BUNDLE=<dir with onnx/model.onnx> KOREDACT_REF=<jsonl: {text, decoded:[[s,e,type,score]], raw:[...]}>
//!   ORT_DYLIB_PATH=<libonnxruntime> cargo test --no-default-features --features load-dynamic --test parity_reference -- --ignored
use std::io::BufRead;

use koredact::Masker;
use serde::Deserialize;

#[derive(Deserialize)]
struct Row { text: String, decoded: Vec<(usize, usize, String, f32)>, raw: Vec<(usize, usize, String, f32)> }

fn spans(v: &[koredact::Span]) -> Vec<(usize, usize, String)> {
    v.iter().map(|s| (s.start, s.end, s.entity.as_str().to_string())).collect()
}
fn refs(v: &[(usize, usize, String, f32)]) -> Vec<(usize, usize, String)> {
    v.iter().map(|(s, e, t, _)| (*s, *e, t.clone())).collect()
}

#[test]
#[ignore]
fn spans_match_python_reference_on_dev() {
    let bundle = std::env::var("KOREDACT_BUNDLE").expect("KOREDACT_BUNDLE");
    let refp = std::env::var("KOREDACT_REF").expect("KOREDACT_REF");
    if let Ok(dylib) = std::env::var("ORT_DYLIB_PATH") { ort::init_from(dylib).expect("dylib").commit(); }
    let mut m = Masker::from_dir(std::path::Path::new(&bundle), 1).expect("bundle");
    let f = std::io::BufReader::new(std::fs::File::open(&refp).expect("ref"));
    let (mut n, mut bad_raw, mut bad_dec, mut max_score_diff) = (0usize, 0usize, 0usize, 0f32);
    for line in f.lines() {
        let row: Row = serde_json::from_str(&line.unwrap()).unwrap();
        let raw = m.predict_raw(&row.text).unwrap();
        let dec = m.predict(&row.text).unwrap();
        if spans(&raw) != refs(&row.raw) { bad_raw += 1; if bad_raw <= 3 { eprintln!("RAW #{n}: {:?}\n  ref {:?}", spans(&raw), refs(&row.raw)); } }
        if spans(&dec) != refs(&row.decoded) { bad_dec += 1; if bad_dec <= 3 { eprintln!("DEC #{n}: {:?}\n  ref {:?}", spans(&dec), refs(&row.decoded)); } }
        for (a, b) in dec.iter().zip(row.decoded.iter()) { if (a.start, a.end) == (b.0, b.1) { max_score_diff = max_score_diff.max((a.score - b.3).abs()); } }
        n += 1;
    }
    eprintln!("docs {n} · raw mismatches {bad_raw} · decoded mismatches {bad_dec} · max score diff {max_score_diff:.2e}");
    assert!(n > 0);
    assert_eq!((bad_raw, bad_dec), (0, 0));
}
