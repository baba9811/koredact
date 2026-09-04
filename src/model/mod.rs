//! ONNX 추론 경로 — 창 분할 · 토크나이즈 · 로짓 · 라벨 그룹화.
//! 파이프라인 내부 전용(`pub(crate)`) — 공개 API 는 `lib.rs` 파사드가 유일한 진입점.
pub(crate) mod infer;
pub(crate) mod label;
pub(crate) mod tokenize;
pub(crate) mod window;
