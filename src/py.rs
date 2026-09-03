//! Python bindings (feature `python`): `_koredact.Masker(dir, ort_dylib=None, threads=1)`.
use std::path::PathBuf;
use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::Masker as Inner;

/// ort's environment is process-global and first-commit-wins: remember the dylib we committed so a later
/// Masker asking for a different libonnxruntime fails loudly instead of silently using the first one.
static ORT_DYLIB: OnceLock<PathBuf> = OnceLock::new();

fn init_ort(path: &str) -> PyResult<()> {
    let want = std::fs::canonicalize(path).map_err(|e| PyValueError::new_err(format!("onnxruntime dylib {path}: {e}")))?;
    if let Some(have) = ORT_DYLIB.get() {
        if *have != want {
            return Err(PyRuntimeError::new_err(format!("onnxruntime already loaded from {}, cannot switch to {}", have.display(), want.display())));
        }
        return Ok(());
    }
    ort::init_from(&want).map_err(|e| PyRuntimeError::new_err(format!("onnxruntime dylib {}: {e}", want.display())))?.commit();
    let _ = ORT_DYLIB.set(want);
    Ok(())
}

#[pyclass(name = "Masker")]
struct PyMasker { inner: Inner }

fn err(e: crate::Error) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

#[pymethods]
impl PyMasker {
    #[new]
    #[pyo3(signature = (dir, ort_dylib=None, threads=1))]
    fn new(dir: String, ort_dylib: Option<String>, threads: usize) -> PyResult<Self> {
        if threads == 0 { return Err(PyValueError::new_err("threads must be >= 1")); }
        if let Some(p) = ort_dylib { init_ort(&p)?; }   // load-dynamic: libonnxruntime from the `onnxruntime` wheel
        Ok(PyMasker { inner: Inner::from_dir(std::path::Path::new(&dir), threads).map_err(err)? })
    }

    /// [(start, end, type, score)] after the decoder. Inference runs with the GIL released; `&mut self`
    /// serializes calls on one Masker (ORT intra-op threads parallelize inside a call).
    fn predict(&mut self, py: Python<'_>, text: String) -> PyResult<Vec<(usize, usize, String, f32)>> {
        let inner = &mut self.inner;
        let spans = py.detach(move || inner.predict(&text)).map_err(err)?;
        Ok(spans.into_iter().map(|s| (s.start, s.end, s.entity.as_str().to_string(), s.score)).collect())
    }

    /// Raw model spans (decoder not applied).
    fn predict_raw(&mut self, py: Python<'_>, text: String) -> PyResult<Vec<(usize, usize, String, f32)>> {
        let inner = &mut self.inner;
        let spans = py.detach(move || inner.predict_raw(&text)).map_err(err)?;
        Ok(spans.into_iter().map(|s| (s.start, s.end, s.entity.as_str().to_string(), s.score)).collect())
    }

    fn mask(&mut self, py: Python<'_>, text: String) -> PyResult<String> {
        let inner = &mut self.inner;
        py.detach(move || inner.mask(&text)).map_err(err)
    }
}

#[pymodule]
fn _koredact(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMasker>()?;
    m.add("DECODER_VERSION", crate::DECODER_VERSION)?;
    Ok(())
}
