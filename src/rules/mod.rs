//! 결정적 후처리 — 디코더 규칙 · 정규식 백스톱 · 마스킹 렌더.
//! `decode` 만 크레이트 최상위로 재노출됨(lib.rs) — 디코더 버전·벡터가 배포 계약이라 외부 대사 필요.
pub mod decode;

pub(crate) mod backstop;
pub(crate) mod mask;
