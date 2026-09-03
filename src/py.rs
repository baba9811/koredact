//! Python bindings (feature `python`): `_koredact.Masker(dir, ort_dylib=None, threads=1)`.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::Masker as Inner;

#[pyclass(name = "Masker")]
struct PyMasker { inner: Inner }

fn err(e: crate::Error) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

#[pymethods]
impl PyMasker {
    #[new]
    #[pyo3(signature = (dir, ort_dylib=None, threads=1))]
    fn new(dir: String, ort_dylib: Option<String>, threads: usize) -> PyResult<Self> {
        if let Some(p) = ort_dylib {
            // load-dynamic: point ort at the libonnxruntime shipped by the `onnxruntime` wheel; only the
            // first commit in a process takes effect (later calls are no-ops by ort's contract)
            ort::init_from(&p).map_err(|e| PyRuntimeError::new_err(format!("onnxruntime dylib {p}: {e}")))?.commit();
        }
        Ok(PyMasker { inner: Inner::from_dir(std::path::Path::new(&dir), threads).map_err(err)? })
    }

    /// [(start, end, type, score)] after the decoder.
    fn predict(&mut self, text: &str) -> PyResult<Vec<(usize, usize, String, f32)>> {
        Ok(self.inner.predict(text).map_err(err)?.into_iter().map(|s| (s.start, s.end, s.entity.as_str().to_string(), s.score)).collect())
    }

    /// Raw model spans (decoder not applied).
    fn predict_raw(&mut self, text: &str) -> PyResult<Vec<(usize, usize, String, f32)>> {
        Ok(self.inner.predict_raw(text).map_err(err)?.into_iter().map(|s| (s.start, s.end, s.entity.as_str().to_string(), s.score)).collect())
    }

    fn mask(&mut self, text: &str) -> PyResult<String> { self.inner.mask(text).map_err(err) }
}

#[pymodule]
fn _koredact(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMasker>()?;
    m.add("DECODER_VERSION", crate::DECODER_VERSION)?;
    Ok(())
}
