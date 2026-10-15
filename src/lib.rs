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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde::Deserialize;

pub use decode::DECODER_VERSION;
pub use error::Error;
pub use rules::mask::MaskTokens;
pub use types::{EntityType, Span};

#[derive(Deserialize)]
struct TokConfig { #[serde(default = "default_max_len")] model_max_length: usize }
fn default_max_len() -> usize { 512 }

pub struct Masker { tok: model::tokenize::Tok, labels: model::label::Labels, model: model::infer::Model, mask_tokens: MaskTokens, dir: PathBuf, threads: usize, preference: String, pool: Vec<Masker> }

impl Masker {
    /// Load a published bundle dir: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model.onnx`.
    pub fn from_dir(dir: &Path, threads: usize) -> Result<Masker, Error> {
        Self::from_dir_with_device(dir, threads, "auto")
    }

    /// Prefer an accelerator when available; recoverable accelerator errors fall back to CPU.
    pub fn from_dir_with_device(dir: &Path, threads: usize, device: &str) -> Result<Masker, Error> {
        model::device::choices(device)?;
        let dir = dir.canonicalize()?;
        let need = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"];
        for n in need {
            if !dir.join(n).is_file() { return Err(Error::Bundle(format!("missing {n} in {}", dir.display()))); }
        }
        let tc: TokConfig = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)?;
        let labels = model::label::Labels::load(&dir.join("config.json"))?;
        let model = model::infer::Model::load(&dir.join("onnx/model.onnx"), labels.0.len(), threads, device)?;
        let tok = model::tokenize::Tok::load(&dir.join("tokenizer.json"), tc.model_max_length)?;
        Ok(Masker { tok, labels, model, mask_tokens: MaskTokens::default(), dir, threads, preference: device.to_owned(), pool: Vec::new() })
    }

    /// Selected provider for the single-text session; unsupported graph operators may still run on CPU.
    pub fn device(&self) -> &'static str { self.model.device() }
    pub fn fallback_reason(&self) -> Option<&str> { self.model.fallback_reason() }

    /// Ordered batch. Each worker owns a session, loaded on first use and kept for this Masker's lifetime;
    /// `workers` is an upper bound, not a CPU-core autodetection setting.
    pub fn predict_many(&mut self, texts: &[String], workers: usize, keep: Option<&[EntityType]>, backstop: bool) -> Result<Vec<Vec<Span>>, Error> {
        self.map_many(texts, workers, |masker, text| masker.predict_opts(text, keep, backstop))
    }

    pub fn mask_many(&mut self, texts: &[String], workers: usize, keep: Option<&[EntityType]>, backstop: bool) -> Result<Vec<String>, Error> {
        self.map_many(texts, workers, |masker, text| masker.mask_opts(text, keep, backstop))
    }

    fn map_many<T: Send>(&mut self, texts: &[String], workers: usize, operation: impl Fn(&mut Self, &str) -> Result<T, Error> + Sync) -> Result<Vec<T>, Error> {
        if workers == 0 { return Err(Error::Bundle("workers must be >= 1".into())); }
        let extra = workers.min(texts.len()).saturating_sub(1);
        // 추가 세션은 Masker 수명 동안 재사용 · 최대 사용 수 유지(축소 없음) · 적재 실패 시 기존 세션만으로 처리
        while self.pool.len() < extra {
            let Ok(replica) = Self::from_dir_with_device(&self.dir, self.threads, &self.preference) else { break };
            self.pool.push(replica);
        }
        let mut pool = std::mem::take(&mut self.pool);
        for replica in &mut pool { replica.mask_tokens = self.mask_tokens.clone(); }
        let used = extra.min(pool.len());
        let output = map_ordered(self, &mut pool[..used], texts, operation);
        // 호출 스레드 패닉 시에만 풀 유실 · 다음 배치 호출에서 재적재
        self.pool = pool;
        output
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

/// 호출 스레드(`first`)와 `rest` 워커 스레드가 다음 미처리 번호를 차례로 가져가는 동적 배분 · 결과는 입력 순서로 반환
fn map_ordered<W: Send, T: Send>(first: &mut W, rest: &mut [W], texts: &[String], operation: impl Fn(&mut W, &str) -> Result<T, Error> + Sync) -> Result<Vec<T>, Error> {
    let (next, failed) = (AtomicUsize::new(0), AtomicBool::new(false));
    // 번호는 증가 순서로만 배분 · 실패 관측 후 새 배분만 중단 → 가장 앞선 실패 입력까지 처리 완료, 그 오류 반환
    let work = |worker: &mut W| {
        let mut done = Vec::new();
        while !failed.load(Ordering::Relaxed) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(text) = texts.get(index) else { break };
            let result = operation(worker, text);
            if result.is_err() { failed.store(true, Ordering::Relaxed); }
            done.push((index, result));
        }
        done
    };
    let mut slots: Vec<Option<Result<T, Error>>> = texts.iter().map(|_| None).collect();
    std::thread::scope(|scope| {
        let work = &work;
        // 스레드 생성 실패 시 생성된 워커와 호출 스레드만으로 처리
        let handles: Vec<_> = rest.iter_mut().map_while(|worker| std::thread::Builder::new().spawn_scoped(scope, move || work(worker)).ok()).collect();
        let mut done = work(first);
        for handle in handles { done.extend(handle.join().unwrap_or_default()); }
        for (index, result) in done { slots[index] = Some(result); }
    });
    // 패닉한 워커의 처리분은 빈 칸 · 그 위치에서 패닉 오류 반환
    slots.into_iter().map(|slot| slot.unwrap_or_else(|| Err(Error::Bundle("batch worker panicked".into())))).collect()
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

    #[test]
    fn batch_keeps_input_order_and_returns_lowest_failing_index() {
        let texts: Vec<String> = (0..40).map(|i| i.to_string()).collect();
        for round in 0..200 {
            let error = map_ordered(&mut (), &mut [(), (), ()], &texts, |_, text| match text {
                // 앞선 실패를 늦게 끝내 뒤쪽 실패가 먼저 관측되는 순서 유도
                "7" => { std::thread::sleep(std::time::Duration::from_millis(2)); Err(Error::Bundle(text.into())) }
                "13" | "31" => Err(Error::Bundle(text.into())),
                _ => Ok(()),
            }).unwrap_err();
            assert_eq!(error.to_string(), "model bundle: 7", "round {round}");
        }
        let numbers = map_ordered(&mut (), &mut [(), ()], &texts, |_, text| {
            if text.ends_with('5') { std::thread::sleep(std::time::Duration::from_millis(1)); }
            Ok(text.parse::<usize>().unwrap())
        }).unwrap();
        assert_eq!(numbers, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn batch_worker_panic_becomes_error() {
        let texts: Vec<String> = (0..8).map(|i| i.to_string()).collect();
        let started = AtomicBool::new(false);
        let result = map_ordered(&mut false, &mut [true], &texts, |worker, _| {
            // 호출 스레드는 워커가 입력을 가져갈 때까지 대기 · 패닉은 항상 워커 스레드에서 발생
            if *worker { started.store(true, Ordering::SeqCst); panic!("worker bug"); }
            while !started.load(Ordering::SeqCst) { std::thread::yield_now(); }
            Ok(())
        });
        assert_eq!(result.unwrap_err().to_string(), "model bundle: batch worker panicked");
    }

    fn tiny() -> Masker {
        Masker::from_dir_with_device(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_bundle"), 1, "cpu").unwrap()
    }

    // 반환된 Arc 보유 동안 세션 해제 불가 · 같은 주소 재사용 없이 동일성 비교 가능
    fn sessions(masker: &Masker) -> Vec<std::sync::Arc<ort::session::SharedSessionInner>> {
        masker.pool.iter().map(|replica| replica.model.session_inner()).collect()
    }

    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run with the wheel runtime"]
    fn batch_sessions_load_once_and_match_sequential() {
        let mut masker = tiny();
        let texts: Vec<String> = vec!["홍길동".into(), "문의 010-1234-5678 홍길동님 ".repeat(40), String::new(), "a@b.test 010-1234-5678".into(), "주문 12345678".into()];
        let expected: Vec<Vec<Span>> = texts.iter().map(|text| masker.predict(text).unwrap()).collect();
        assert_eq!(masker.predict_many(&texts, 3, None, false).unwrap(), expected);
        let loaded = sessions(&masker);
        assert_eq!(loaded.len(), 2);
        for workers in [3, 1, 2] {
            assert_eq!(masker.predict_many(&texts, workers, None, false).unwrap(), expected, "workers {workers}");
        }
        let same = |masker: &Masker| sessions(masker).iter().zip(&loaded).all(|(now, before)| std::sync::Arc::ptr_eq(now, before));
        assert!(masker.pool.len() == 2 && same(&masker));
        assert_eq!(masker.predict_many(&texts, 99, None, false).unwrap(), expected);
        assert!(masker.pool.len() == 4 && same(&masker));
        assert!(masker.predict_many(&[], 4, None, false).unwrap().is_empty());
        let error = masker.map_many(&texts, 4, |_, text| if text.is_empty() { Err(Error::Bundle("empty".into())) } else { Ok(()) }).unwrap_err();
        assert_eq!(error.to_string(), "model bundle: empty");
        assert!(masker.pool.len() == 4 && same(&masker));
    }

    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run with the wheel runtime"]
    fn batch_replicas_follow_mask_token_changes() {
        let mut masker = tiny();
        let texts: Vec<String> = (0..8).map(|i| format!("주문 {i}2345678 문의 010-1234-567{i}")).collect();
        let before = masker.mask_many(&texts, 4, None, true).unwrap();
        let mut tokens = MaskTokens::default();
        for entity in EntityType::ALL { tokens.set(entity, "***"); }
        masker.set_mask_tokens(tokens);
        let after = masker.mask_many(&texts, 4, None, true).unwrap();
        assert!(masker.pool.len() == 3 && masker.pool.iter().all(|replica| replica.mask_tokens == masker.mask_tokens));
        assert_eq!(after, texts.iter().map(|text| masker.mask_opts(text, None, true).unwrap()).collect::<Vec<_>>());
        assert_ne!(before, after);
    }
}
