//! koredact — Korean PII masking runtime: BERT token-classification (ONNX) + rule decoder.
//! Pipeline per text: char windows (400/100) → tokenize (char offsets) → ONNX logits (opt-in: margin-gated rerun on
//! a fallback model) → argmax → simple grouping → window merge → decoder v3 → (optional regex backstop) →
//! spans / masked text.
pub mod error;
pub mod types;

mod model;
mod rules;

// 디코더 버전·벡터는 배포 계약 — 경로를 내부 묶음(`rules`)에 묶지 않고 크레이트 최상위로 재노출
pub use rules::decode;

#[cfg(feature = "python")]
mod py;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use serde::Deserialize;

pub use decode::DECODER_VERSION;
pub use error::Error;
pub use rules::mask::MaskTokens;
pub use types::{EntityType, Span};

#[derive(Deserialize)]
struct TokConfig { #[serde(default = "default_max_len")] model_max_length: usize }
fn default_max_len() -> usize { 512 }

pub struct Masker { tok: model::tokenize::Tok, labels: model::label::Labels, model: model::infer::Model, mask_tokens: MaskTokens, dir: PathBuf, threads: usize, preference: String, pool: Vec<Masker>, fallback: Option<model::cascade::Fallback> }

impl Masker {
    /// Load a published bundle dir: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model.onnx`.
    pub fn from_dir(dir: &Path, threads: usize) -> Result<Masker, Error> {
        Self::from_dir_with_device(dir, threads, "auto")
    }

    /// Prefer an accelerator when available; recoverable accelerator errors fall back to CPU.
    pub fn from_dir_with_device(dir: &Path, threads: usize, device: &str) -> Result<Masker, Error> {
        model::device::choices(device)?;
        let dir = bundle_dir(dir)?;
        let tc: TokConfig = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)?;
        let labels = model::label::Labels::load(&dir.join("config.json"))?;
        let model = model::infer::Model::load(&dir.join("onnx/model.onnx"), labels.0.len(), threads, device)?;
        let tok = model::tokenize::Tok::load(&dir.join("tokenizer.json"), tc.model_max_length)?;
        Ok(Masker { tok, labels, model, mask_tokens: MaskTokens::default(), dir, threads, preference: device.to_owned(), pool: Vec::new(), fallback: None })
    }

    /// Opt-in margin-gated cascade. Each window runs on this Masker's model first; when the smallest top-1 minus top-2
    /// logit gap over its non-special tokens is below `margin`, the window is rerun on the `dir` bundle and that model's
    /// logits replace the whole window. The bundle must carry the same label map and byte-identical tokenizer files; its
    /// ONNX graph and non-label `config.json` fields may differ. It loads with this Masker's threads and device preference.
    pub fn with_fallback(self, dir: &Path, margin: f32) -> Result<Masker, Error> {
        self.attach_fallback(dir, margin, Arc::default())
    }

    /// `(windows, windows rerun on the fallback)` since `with_fallback`, batch sessions included; `(0, 0)` without one.
    pub fn fallback_stats(&self) -> (u64, u64) {
        self.fallback.as_ref().map_or((0, 0), |fallback| fallback.counts.get())
    }

    fn attach_fallback(mut self, dir: &Path, margin: f32, counts: Arc<model::cascade::Counts>) -> Result<Masker, Error> {
        let margin = model::cascade::check_margin(margin)?;
        let dir = fallback_dir(&self.dir, &self.labels, dir)?;
        let model = model::infer::Model::load(&dir.join("onnx/model.onnx"), self.labels.0.len(), self.threads, &self.preference)?;
        self.fallback = Some(model::cascade::Fallback { dir, margin, counts, model });
        // 기존 추가 세션은 대체 모델 없음 · 다음 배치 호출에서 대체 모델과 함께 재적재
        self.pool.clear();
        Ok(self)
    }

    /// 같은 번들·`threads`·장치 선호·대체 모델로 적재 · 대체 카운터 공유
    fn replica(&self) -> Result<Masker, Error> {
        let replica = Self::from_dir_with_device(&self.dir, self.threads, &self.preference)?;
        match &self.fallback {
            Some(fallback) => replica.attach_fallback(&fallback.dir, fallback.margin, fallback.counts.clone()),
            None => Ok(replica),
        }
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
            let Ok(replica) = self.replica() else { break };
            self.pool.push(replica);
        }
        let mut pool = std::mem::take(&mut self.pool);
        for replica in &mut pool { replica.mask_tokens = self.mask_tokens.clone(); }
        let used = extra.min(pool.len());
        let (output, panicked) = map_ordered(self, &mut pool[..used], texts, operation);
        // 패닉한 추가 세션은 상태 보장 불가 · 풀에서 제거, 다음 호출에서 새로 적재
        for position in panicked.into_iter().rev() { pool.remove(position); }
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
            let mut rows = self.model.logits(enc.get_ids(), enc.get_type_ids(), enc.get_attention_mask())?;
            if let Some(fallback) = &mut self.fallback { rows = fallback.choose(rows, &enc)?; }
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

/// Canonical bundle dir with `config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model.onnx`.
fn bundle_dir(dir: &Path) -> Result<PathBuf, Error> {
    let dir = dir.canonicalize()?;
    for n in ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"] {
        if !dir.join(n).is_file() { return Err(Error::Bundle(format!("missing {n} in {}", dir.display()))); }
    }
    Ok(dir)
}

/// 대체 번들 검사 · 라벨 맵과 토크나이저 파일 바이트가 기본 번들과 같아야 함, ONNX 그래프와 라벨 외 `config.json` 필드는 차이 허용
/// 토큰화·라벨이 다르면 같은 창의 로짓 행이 서로 다른 토큰·라벨을 가리킴
fn fallback_dir(primary: &Path, labels: &model::label::Labels, dir: &Path) -> Result<PathBuf, Error> {
    let dir = bundle_dir(dir)?;
    for file in ["tokenizer.json", "tokenizer_config.json"] {
        if std::fs::read(primary.join(file))? != std::fs::read(dir.join(file))? {
            return Err(Error::Bundle(format!("fallback {file} differs from {}", primary.display())));
        }
    }
    if model::label::Labels::load(&dir.join("config.json"))? != *labels {
        return Err(Error::Bundle(format!("fallback config.json label map differs from {}", primary.display())));
    }
    Ok(dir)
}

/// 호출 스레드(`first`)와 `rest` 워커 스레드가 다음 미처리 번호를 차례로 가져가는 동적 배분 · 결과는 입력 순서로 반환
/// 두 번째 값: 패닉한 `rest` 워커의 위치
fn map_ordered<W: Send, T: Send>(first: &mut W, rest: &mut [W], texts: &[String], operation: impl Fn(&mut W, &str) -> Result<T, Error> + Sync) -> (Result<Vec<T>, Error>, Vec<usize>) {
    let (next, failed) = (AtomicUsize::new(0), AtomicBool::new(false));
    let worker_panic = || Error::Bundle("batch worker panicked".into());
    // 번호는 증가 순서로만 배분 · 실패 관측 후 새 배분만 중단 → 가장 앞선 실패 입력까지 처리 완료, 그 오류 반환
    // 워커 스레드의 패닉은 해당 번호의 실패로 기록, 완료 결과 유지 · 호출 스레드 패닉은 그대로 전파
    let work = |worker: &mut W, isolate: bool| {
        let (mut done, mut panicked) = (Vec::new(), false);
        while !failed.load(Ordering::Relaxed) {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(text) = texts.get(index) else { break };
            let result = if isolate {
                catch_unwind(AssertUnwindSafe(|| operation(worker, text))).unwrap_or_else(|_| { panicked = true; Err(worker_panic()) })
            } else {
                operation(worker, text)
            };
            if result.is_err() { failed.store(true, Ordering::Relaxed); }
            done.push((index, result));
        }
        (done, panicked)
    };
    let mut slots: Vec<Option<Result<T, Error>>> = texts.iter().map(|_| None).collect();
    let mut panicked = Vec::new();
    std::thread::scope(|scope| {
        let work = &work;
        // 스레드 생성 실패 시 생성된 워커와 호출 스레드만으로 처리
        let handles: Vec<_> = rest.iter_mut().map_while(|worker| std::thread::Builder::new().spawn_scoped(scope, move || work(worker, true)).ok()).collect();
        let mut done = work(first, false).0;
        for (position, handle) in handles.into_iter().enumerate() {
            let (results, worker_panicked) = handle.join().unwrap_or((Vec::new(), true));
            if worker_panicked { panicked.push(position); }
            done.extend(results);
        }
        for (index, result) in done { slots[index] = Some(result); }
    });
    // 빈 칸은 첫 실패 뒤 미배분 번호라 순서대로 모으면 그 앞 실패에서 중단 · 그 외 빈 칸은 패닉 오류로 대체
    (slots.into_iter().map(|slot| slot.unwrap_or_else(|| Err(worker_panic()))).collect(), panicked)
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
            }).0.unwrap_err();
            assert_eq!(error.to_string(), "model bundle: 7", "round {round}");
        }
        let numbers = map_ordered(&mut (), &mut [(), ()], &texts, |_, text| {
            if text.ends_with('5') { std::thread::sleep(std::time::Duration::from_millis(1)); }
            Ok(text.parse::<usize>().unwrap())
        }).0.unwrap();
        assert_eq!(numbers, (0..40).collect::<Vec<_>>());
    }

    #[test]
    fn batch_worker_panic_becomes_error() {
        let texts: Vec<String> = (0..8).map(|i| i.to_string()).collect();
        let started = AtomicBool::new(false);
        let (result, panicked) = map_ordered(&mut false, &mut [true], &texts, |worker, _| {
            // 호출 스레드는 워커가 입력을 가져갈 때까지 대기 · 패닉은 항상 워커 스레드에서 발생
            if *worker { started.store(true, Ordering::SeqCst); panic!("worker bug"); }
            wait_for(&started);
            Ok(())
        });
        assert_eq!(result.unwrap_err().to_string(), "model bundle: batch worker panicked");
        assert_eq!(panicked, [0]);
    }

    // 최대 10초 대기 · 상대 스레드가 단계에 오지 못하면 무한 대기 대신 검사 실패
    fn wait_for(flag: &AtomicBool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !flag.load(Ordering::SeqCst) {
            assert!(std::time::Instant::now() < deadline, "peer thread never reached the expected step");
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }

    #[test]
    fn worker_panic_keeps_completed_results_behind_lowest_error() {
        // 순서 강제: 워커가 앞 번호 처리 완료 → 호출 스레드가 그 뒤 번호에서 느린 오류 → 워커가 더 뒤 번호에서 패닉
        // 워커의 완료 결과 유실 시 그 빈 칸이 호출 스레드 오류보다 앞선 패닉 오류로 반환됨
        let texts: Vec<String> = (0..4).map(|i| i.to_string()).collect();
        let [worker_done, caller_failing, worker_panicking] = [(); 3].map(|_| AtomicBool::new(false));
        let worker_calls = AtomicUsize::new(0);
        let (result, panicked) = map_ordered(&mut false, &mut [true], &texts, |is_worker, text| {
            if *is_worker {
                if worker_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    worker_done.store(true, Ordering::SeqCst);
                    wait_for(&caller_failing);
                    return Ok(());
                }
                worker_panicking.store(true, Ordering::SeqCst);
                panic!("worker bug");
            }
            if text == "0" { wait_for(&worker_done); return Ok(()); }
            caller_failing.store(true, Ordering::SeqCst);
            wait_for(&worker_panicking);
            Err(Error::Bundle(format!("caller {text}")))
        });
        assert!(result.unwrap_err().to_string().starts_with("model bundle: caller "));
        assert_eq!(panicked, [0]);
    }

    fn tiny_dir() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_bundle") }

    fn tiny() -> Masker { Masker::from_dir_with_device(&tiny_dir(), 1, "cpu").unwrap() }

    /// 임시 디렉터리의 소형 번들 사본 · `edit`이 사본 파일 변경
    fn tiny_copy(name: &str, edit: impl FnOnce(&Path)) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("koredact-{name}-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("onnx")).unwrap();
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"] {
            std::fs::copy(tiny_dir().join(file), dir.join(file)).unwrap();
        }
        edit(&dir);
        dir
    }

    fn rewrite(dir: &Path, file: &str, change: impl FnOnce(String) -> String) {
        let before = std::fs::read_to_string(dir.join(file)).unwrap();
        let after = change(before.clone());
        assert_ne!(after, before, "{file} unchanged");
        std::fs::write(dir.join(file), after).unwrap();
    }

    /// `classifier.bias`의 B-NAME(23) 항목을 50으로 고정 · 라벨·토크나이저는 같고 출력만 다른 모델
    fn biased(dir: &Path) {
        let path = dir.join("onnx/model.onnx");
        let mut onnx = std::fs::read(&path).unwrap();
        // TensorProto name(8) "classifier.bias" 바로 뒤 raw_data(9) 108바이트 = f32 27개
        let tag = b"\x42\x0fclassifier.bias\x4a\x6c";
        let at = onnx.windows(tag.len()).position(|w| w == tag).expect("classifier.bias raw_data") + tag.len() + 23 * 4;
        onnx[at..at + 4].copy_from_slice(&50f32.to_le_bytes());
        std::fs::write(path, onnx).unwrap();
    }

    #[test]
    fn fallback_bundle_must_share_labels_and_tokenizer_files() {
        let labels = model::label::Labels::load(&tiny_dir().join("config.json")).unwrap();
        let check = |dir: &Path| fallback_dir(&tiny_dir(), &labels, dir).map_err(|e| e.to_string());
        let copies = [
            tiny_copy("fallback-onnx", biased),
            tiny_copy("fallback-config", |dir| rewrite(dir, "config.json", |s| s.replace("\"hidden_dropout_prob\": 0.1", "\"hidden_dropout_prob\": 0.2"))),
            tiny_copy("fallback-labels", |dir| rewrite(dir, "config.json", |s| s.replace("\"23\": \"B-NAME\"", "\"23\": \"B-PHONE\""))),
            tiny_copy("fallback-tokenizer", |dir| rewrite(dir, "tokenizer.json", |s| s + "\n")),
            tiny_copy("fallback-tokenizer-config", |dir| rewrite(dir, "tokenizer_config.json", |s| s.replace("512", "256"))),
            tiny_copy("fallback-no-onnx", |dir| std::fs::remove_file(dir.join("onnx/model.onnx")).unwrap()),
        ];
        // ONNX 그래프와 라벨 외 설정 차이만 허용
        assert_eq!(check(&copies[0]).unwrap(), copies[0].canonicalize().unwrap());
        assert!(check(&copies[1]).is_ok());
        assert!(check(&copies[2]).unwrap_err().contains("config.json label map differs"));
        assert!(check(&copies[3]).unwrap_err().contains("fallback tokenizer.json differs"));
        assert!(check(&copies[4]).unwrap_err().contains("fallback tokenizer_config.json differs"));
        assert!(check(&copies[5]).unwrap_err().contains("missing onnx/model.onnx"));
        for dir in copies { std::fs::remove_dir_all(dir).unwrap(); }
    }

    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run with the wheel runtime"]
    fn cascade_takes_the_fallback_logits_for_unsure_windows() {
        let biased_dir = tiny_copy("cascade-biased", biased);
        let long = "문의 010-1234-5678 홍길동님 ".repeat(40);
        let texts = ["홍길동", long.as_str(), "a@b.test 010-1234-5678"];
        let windows: u64 = texts.iter().map(|text| model::window::starts(text.chars().count()).len() as u64).sum();
        let (mut fast, mut exact) = (tiny(), Masker::from_dir_with_device(&biased_dir, 1, "cpu").unwrap());
        // 모든 창 재실행: 대체 모델 출력과 동일
        let mut always = tiny().with_fallback(&biased_dir, f32::MAX).unwrap();
        for text in texts {
            let want = exact.predict_raw(text).unwrap();
            assert_ne!(fast.predict_raw(text).unwrap(), want);
            assert_eq!(always.predict_raw(text).unwrap(), want);
            assert_eq!(always.predict(text).unwrap(), exact.predict(text).unwrap());
        }
        assert_eq!(always.fallback_stats(), (2 * windows, 2 * windows));
        assert_eq!((fast.fallback_stats(), exact.fallback_stats()), ((0, 0), (0, 0)));
        // 같은 번들이 대체 모델이면 마진과 무관하게 기본 출력 유지
        for margin in [f32::MIN_POSITIVE, 0.05, 1.0, f32::MAX] {
            let mut same = tiny().with_fallback(&tiny_dir(), margin).unwrap();
            for text in texts { assert_eq!(same.predict_raw(text).unwrap(), fast.predict_raw(text).unwrap(), "margin {margin}"); }
            let (seen, rerun) = same.fallback_stats();
            assert!(seen == windows && rerun <= seen && (margin < f32::MAX || rerun == windows), "margin {margin}: {seen} {rerun}");
        }
        for margin in [0.0, -1.0, f32::NAN, f32::INFINITY] { assert!(tiny().with_fallback(&tiny_dir(), margin).is_err()); }
        let tokenizer = tiny_copy("cascade-tokenizer", |dir| rewrite(dir, "tokenizer.json", |s| s + "\n"));
        assert!(tiny().with_fallback(&tokenizer, 1.0).err().unwrap().to_string().contains("tokenizer.json differs"));
        for dir in [biased_dir, tokenizer] { std::fs::remove_dir_all(dir).unwrap(); }
    }

    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run with the wheel runtime"]
    fn batch_replicas_carry_the_fallback_and_share_its_counters() {
        let biased_dir = tiny_copy("cascade-replicas", biased);
        let texts: Vec<String> = vec!["홍길동".into(), "문의 010-1234-5678 홍길동님 ".repeat(40), String::new(), "a@b.test 010-1234-5678".into(), "주문 12345678".into()];
        let mut exact = Masker::from_dir_with_device(&biased_dir, 1, "cpu").unwrap();
        let expected: Vec<Vec<Span>> = texts.iter().map(|text| exact.predict(text).unwrap()).collect();
        let mut masker = tiny();
        // 대체 모델 없는 추가 세션 2개 적재 후 대체 모델 추가 → 기존 추가 세션 폐기
        assert_ne!(masker.predict_many(&texts, 3, None, false).unwrap(), expected);
        let mut masker = masker.with_fallback(&biased_dir, f32::MAX).unwrap();
        assert!(masker.pool.is_empty());
        assert_eq!(texts.iter().map(|text| masker.predict(text).unwrap()).collect::<Vec<_>>(), expected);
        let (windows, rerun) = masker.fallback_stats();
        // 빈 입력의 창은 비특수 토큰이 없어 재실행 없음
        assert!(rerun > 0 && rerun < windows);
        for workers in [3, 2] {
            assert_eq!(masker.predict_many(&texts, workers, None, false).unwrap(), expected, "workers {workers}");
        }
        let dir = biased_dir.canonicalize().unwrap();
        assert!(masker.pool.len() == 2 && masker.pool.iter().all(|replica| replica.fallback.as_ref().is_some_and(|f| f.dir == dir && f.margin == f32::MAX)));
        assert_eq!(masker.fallback_stats(), (3 * windows, 3 * rerun));
        std::fs::remove_dir_all(biased_dir).unwrap();
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
        let (before, main, replica_panicked) = (sessions(&masker), &masker as *const Masker as usize, AtomicBool::new(false));
        let error = masker.map_many(&texts, 4, |worker, _| {
            if worker as *const Masker as usize != main { replica_panicked.store(true, Ordering::SeqCst); panic!("replica bug"); }
            wait_for(&replica_panicked);
            Ok(())
        }).unwrap_err();
        assert_eq!(error.to_string(), "model bundle: batch worker panicked");
        assert!(masker.pool.len() < 4);
        assert_eq!(masker.predict_many(&texts, 99, None, false).unwrap(), expected);
        let kept = sessions(&masker).iter().filter(|now| before.iter().any(|old| std::sync::Arc::ptr_eq(now, old))).count();
        assert!(masker.pool.len() == 4 && kept < 4);
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
