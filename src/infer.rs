//! ONNX session: (input_ids, token_type_ids, attention_mask)[1, seq] i64 → logits [1, seq, n_labels].
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::error::Error;

pub struct Model { session: Session, pub n_labels: usize }

impl Model {
    pub fn load(onnx: &Path, n_labels: usize, threads: usize) -> Result<Model, Error> {
        let session = Session::builder()?
            .with_intra_threads(threads).map_err(|e| Error::Bundle(format!("session options: {e}")))?
            .commit_from_file(onnx)?;
        Ok(Model { session, n_labels })
    }

    /// Returns per-token logits rows (seq × n_labels).
    pub fn logits(&mut self, ids: &[u32], type_ids: &[u32], mask: &[u32]) -> Result<Vec<Vec<f32>>, Error> {
        let seq = ids.len() as i64;
        let to_i64 = |v: &[u32]| v.iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, seq], to_i64(ids)))?,
            "token_type_ids" => Tensor::from_array((vec![1i64, seq], to_i64(type_ids)))?,
            "attention_mask" => Tensor::from_array((vec![1i64, seq], to_i64(mask)))?,
        ])?;
        let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != seq || dims[2] as usize != self.n_labels {
            return Err(Error::Bundle(format!("logits shape {dims:?} != [1, {seq}, {}]", self.n_labels)));
        }
        Ok(data.chunks(self.n_labels).map(|r| r.to_vec()).collect())
    }
}
