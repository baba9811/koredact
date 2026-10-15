//! 마진 게이트 캐스케이드: 빠른 모델의 top-1·top-2 로짓 차가 `margin` 미만인 창만 정밀 모델로 재실행.
//! 재실행 창은 정밀 모델 로짓이 창 전체를 대체 · 특수 토큰은 예측에서 빠지므로 판정에서도 제외.
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use tokenizers::Encoding;

use super::infer::Model;
use crate::error::Error;

/// Windows seen and windows rerun on the exact model; one instance is shared by a Masker and its batch replicas.
#[derive(Default)]
pub struct Counts { windows: AtomicU64, fallback: AtomicU64 }

impl Counts {
    pub fn get(&self) -> (u64, u64) { (self.windows.load(Relaxed), self.fallback.load(Relaxed)) }
}

/// Exact model of the cascade. `dir` and `margin` let the facade rebuild it for batch replicas.
pub struct Fallback { pub dir: PathBuf, pub margin: f32, pub counts: Arc<Counts>, pub model: Model }

impl Fallback {
    /// `fast` logits, or the exact model's logits for the whole window when a non-special token is unsure.
    pub fn choose(&mut self, fast: Vec<Vec<f32>>, enc: &Encoding) -> Result<Vec<Vec<f32>>, Error> {
        self.counts.windows.fetch_add(1, Relaxed);
        if !unsure(&fast, enc.get_special_tokens_mask(), self.margin) { return Ok(fast); }
        let exact = self.model.logits(enc.get_ids(), enc.get_type_ids(), enc.get_attention_mask())?;
        self.counts.fallback.fetch_add(1, Relaxed);
        Ok(exact)
    }
}

pub fn check_margin(margin: f32) -> Result<f32, Error> {
    if margin > 0.0 && margin.is_finite() { Ok(margin) } else { Err(Error::Bundle(format!("fallback margin must be finite and > 0, got {margin}"))) }
}

/// 비특수 토큰 중 top-1·top-2 로짓 차가 `margin` 미만인 토큰 존재 여부 · 동점은 차 0 · 라벨 1개면 차 무한대
fn unsure(rows: &[Vec<f32>], special: &[u32], margin: f32) -> bool {
    rows.iter().zip(special).any(|(row, &mark)| mark == 0 && gap(row) < margin)
}

fn gap(row: &[f32]) -> f32 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &value in row {
        if value > first { second = first; first = value; } else if value > second { second = value; }
    }
    first - second
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_tokens_never_trigger_the_fallback() {
        // 특수 토큰 행은 동점(차 0) · 비특수 토큰 차 5와 3
        let rows = vec![vec![1.0, 1.0], vec![5.0, 0.0], vec![-1.0, 2.0], vec![0.5, 0.5]];
        assert!(!unsure(&rows, &[1, 0, 0, 1], 3.0));
        assert!(unsure(&rows, &[1, 0, 0, 1], 3.5));
        assert!(unsure(&rows, &[0, 0, 0, 1], 0.1));
        assert!(!unsure(&rows, &[1, 1, 1, 1], f32::MAX));
        assert!(!unsure(&[], &[], f32::MAX));
    }

    #[test]
    fn tie_is_a_zero_gap_and_one_label_never_is_unsure() {
        assert_eq!(gap(&[2.0, 2.0, 1.0]), 0.0);
        assert_eq!(gap(&[0.0, 4.0, 1.5, 4.0]), 0.0);
        assert!(unsure(&[vec![2.0, 2.0, 1.0]], &[0], f32::MIN_POSITIVE));
        assert_eq!(gap(&[-3.0, 1.0, 0.25]), 0.75);
        assert_eq!(gap(&[7.0]), f32::INFINITY);
        assert!(!unsure(&[vec![7.0]], &[0], f32::MAX));
    }

    #[test]
    fn margin_equal_to_tau_keeps_the_fast_logits() {
        let rows = vec![vec![3.0, 0.5, -1.0]];
        assert_eq!(gap(&rows[0]), 2.5);
        assert!(!unsure(&rows, &[0], 2.5));
        assert!(unsure(&rows, &[0], 2.5f32.next_up()));
        assert!(!unsure(&rows, &[0], 2.5f32.next_down()));
    }

    #[test]
    fn margin_must_be_positive_and_finite() {
        for bad in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_margin(bad).is_err(), "{bad}");
        }
        for good in [f32::MIN_POSITIVE, 2.5, f32::MAX] {
            assert_eq!(check_margin(good).unwrap(), good);
        }
    }
}
