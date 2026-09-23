//! ONNX session: (input_ids, token_type_ids, attention_mask)[1, seq] i64 → logits [1, seq, n_labels].
use std::path::{Path, PathBuf};
use super::device::{self, Device};

use ort::session::Session;
use ort::value::Tensor;

use crate::error::Error;

pub struct Model { session: Option<Session>, n_labels: usize, wants_type_ids: bool, onnx: PathBuf, threads: usize, device: Device, fallback_reason: Option<String> }

impl Model {
    pub fn load(onnx: &Path, n_labels: usize, threads: usize, preference: &str) -> Result<Model, Error> {
        if threads == 0 { return Err(Error::Bundle("threads must be >= 1".into())); }
        if n_labels == 0 { return Err(Error::Bundle("n_labels must be >= 1".into())); }
        let (session, device, fallback_reason) = device::select(preference, |d| device::session(onnx, threads, d))?;
        // graph inputs decide what we feed: BERT exports carry input_ids/attention_mask and usually token_type_ids
        // (an exporter may prune it when the graph never reads it — then feeding it is an ORT error)
        let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        for need in ["input_ids", "attention_mask"] {
            if !names.iter().any(|n| n == need) { return Err(Error::Bundle(format!("onnx graph lacks input {need}: {names:?}"))); }
        }
        let wants_type_ids = names.iter().any(|n| n == "token_type_ids");
        if names.len() != 2 + wants_type_ids as usize {
            return Err(Error::Bundle(format!("unexpected onnx inputs {names:?}")));
        }
        let outs: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
        if !outs.iter().any(|n| n == "logits") {
            return Err(Error::Bundle(format!("onnx graph lacks output logits: {outs:?}")));
        }
        Ok(Model { session: Some(session), n_labels, wants_type_ids, onnx: onnx.to_owned(), threads, device, fallback_reason })
    }

    pub fn device(&self) -> &'static str { self.device.name() }
    pub fn fallback_reason(&self) -> Option<&str> { self.fallback_reason.as_deref() }

    /// Returns per-token logits rows (seq × n_labels).
    pub fn logits(&mut self, ids: &[u32], type_ids: &[u32], mask: &[u32]) -> Result<Vec<Vec<f32>>, Error> {
        if ids.is_empty() || mask.len() != ids.len() || (self.wants_type_ids && type_ids.len() != ids.len()) {
            return Err(Error::Bundle(format!("encoding lengths differ: ids {} mask {} type_ids {}", ids.len(), mask.len(), type_ids.len())));
        }
        match self.run(ids, type_ids, mask) {
            Err(error) if self.device != Device::Cpu => {
                self.fallback_reason = Some(format!("{} inference failed; retrying CPU: {error}", self.device.name()));
                // 실패한 가속기 세션 해제 후 CPU 적재, 두 모델의 동시 메모리 점유 방지
                self.session.take();
                self.device = Device::Cpu;
                self.session = Some(device::session(&self.onnx, self.threads, Device::Cpu)?);
                self.run(ids, type_ids, mask)
            }
            outcome => outcome,
        }
    }

    fn run(&mut self, ids: &[u32], type_ids: &[u32], mask: &[u32]) -> Result<Vec<Vec<f32>>, Error> {
        let seq = ids.len() as i64;
        let to_i64 = |v: &[u32]| v.iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let mut feed = ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, seq], to_i64(ids)))?,
            "attention_mask" => Tensor::from_array((vec![1i64, seq], to_i64(mask)))?,
        ];
        if self.wants_type_ids {
            feed.push(("token_type_ids".into(), Tensor::from_array((vec![1i64, seq], to_i64(type_ids)))?.into()));
        }
        let outputs = self.session.as_mut().ok_or_else(|| Error::Bundle("session unavailable".into()))?.run(feed)?;
        let logits = outputs.get("logits").ok_or_else(|| Error::Bundle("onnx run returned no logits".into()))?;
        let (shape, data) = logits.try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        let expect_len = ids.len().checked_mul(self.n_labels)
            .ok_or_else(|| Error::Bundle(format!("logits size overflow: {} x {}", ids.len(), self.n_labels)))?;
        if dims.len() != 3 || dims[0] != 1 || dims[1] != seq || dims[2] as usize != self.n_labels || data.len() != expect_len {
            return Err(Error::Bundle(format!("logits shape {dims:?} / len {} != [1, {seq}, {}]", data.len(), self.n_labels)));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(Error::Bundle("non-finite logits".into()));
        }
        Ok(data.chunks(self.n_labels).map(|r| r.to_vec()).collect())
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;
    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run with the wheel runtime"]
    fn failed_accelerator_run_retries_same_input_on_cpu() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_bundle/onnx/model.onnx");
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path.parent().unwrap().parent().unwrap().join("config.json")).unwrap()).unwrap();
        let mut model = Model::load(&path, config["id2label"].as_object().unwrap().len(), 1, "cpu").unwrap();
        let expected = model.logits(&[2, 3], &[0, 0], &[1, 1]).unwrap();
        model.device = Device::Cuda;
        model.session.take();
        assert_eq!(model.logits(&[2, 3], &[0, 0], &[1, 1]).unwrap(), expected);
        assert_eq!(model.device(), "cpu");
        assert!(model.fallback_reason().unwrap().contains("retrying CPU"));
        assert!(model.logits(&[2], &[0], &[]).is_err());
        let n_labels = model.n_labels;
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let invalid_session = || Session::builder().unwrap().with_intra_threads(1).unwrap()
                .with_initializer("classifier.bias", std::sync::Arc::new(Tensor::from_array((vec![n_labels as i64], vec![value; n_labels])).unwrap().into_dyn())).unwrap()
                .commit_from_file(&path).unwrap();
            model.session = Some(invalid_session());
            assert!(model.logits(&[2, 3], &[0, 0], &[1, 1]).is_err());
            model.session = Some(invalid_session());
            model.device = Device::Cuda;
            assert_eq!(model.logits(&[2, 3], &[0, 0], &[1, 1]).unwrap(), expected);
            assert_eq!(model.device(), "cpu");
            assert!(model.fallback_reason().unwrap().contains("non-finite"));
        }
    }
}
