//! ONNX session: (input_ids, token_type_ids, attention_mask)[1, seq] i64 → logits [1, seq, n_labels].
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::error::Error;

pub struct Model { session: Session, pub n_labels: usize, wants_type_ids: bool }

impl Model {
    pub fn load(onnx: &Path, n_labels: usize, threads: usize) -> Result<Model, Error> {
        if threads == 0 { return Err(Error::Bundle("threads must be >= 1".into())); }
        if n_labels == 0 { return Err(Error::Bundle("n_labels must be >= 1".into())); }
        let session = Session::builder()?
            .with_intra_threads(threads).map_err(|e| Error::Bundle(format!("session options: {e}")))?
            .commit_from_file(onnx)?;
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
        Ok(Model { session, n_labels, wants_type_ids })
    }

    /// Returns per-token logits rows (seq × n_labels).
    pub fn logits(&mut self, ids: &[u32], type_ids: &[u32], mask: &[u32]) -> Result<Vec<Vec<f32>>, Error> {
        let seq = ids.len() as i64;
        let to_i64 = |v: &[u32]| v.iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let mut feed = ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, seq], to_i64(ids)))?,
            "attention_mask" => Tensor::from_array((vec![1i64, seq], to_i64(mask)))?,
        ];
        if self.wants_type_ids {
            feed.push(("token_type_ids".into(), Tensor::from_array((vec![1i64, seq], to_i64(type_ids)))?.into()));
        }
        let outputs = self.session.run(feed)?;
        let logits = outputs.get("logits").ok_or_else(|| Error::Bundle("onnx run returned no logits".into()))?;
        let (shape, data) = logits.try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != seq || dims[2] as usize != self.n_labels {
            return Err(Error::Bundle(format!("logits shape {dims:?} != [1, {seq}, {}]", self.n_labels)));
        }
        Ok(data.chunks(self.n_labels).map(|r| r.to_vec()).collect())
    }
}
