//! Python bindings (feature `python`): `_koredact.Masker(dir, ort_dylib=None, threads=1)`.
use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use std::collections::HashMap;

use crate::{EntityType, MaskTokens, Masker as Inner};

/// ort's environment is process-global and first-commit-wins: remember the dylib we committed so a later
/// Masker asking for a different libonnxruntime fails loudly instead of silently using the first one.
static ORT_DYLIB: Mutex<Option<PathBuf>> = Mutex::new(None);

fn init_ort(path: &str) -> PyResult<()> {
    let want = std::fs::canonicalize(path).map_err(|e| PyValueError::new_err(format!("onnxruntime dylib {path}: {e}")))?;
    let mut have = ORT_DYLIB.lock().map_err(|_| PyRuntimeError::new_err("ort init lock poisoned"))?;   // check+commit+set 직렬화
    if let Some(h) = have.as_ref() {
        if *h != want {
            return Err(PyRuntimeError::new_err(format!("onnxruntime already loaded from {}, cannot switch to {}", h.display(), want.display())));
        }
        return Ok(());
    }
    let builder = ort::init_from(&want).map_err(|e| PyRuntimeError::new_err(format!("onnxruntime dylib {}: {e}", want.display())))?;
    // commit() -> bool: false = an ort environment was already configured elsewhere in this process (not by us)
    if !builder.commit() {
        return Err(PyRuntimeError::new_err("ort environment already initialized outside koredact; cannot bind onnxruntime dylib"));
    }
    *have = Some(want);
    Ok(())
}

#[pyclass(name = "Masker")]
struct PyMasker { inner: Inner }

fn err(e: crate::Error) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

/// Per-type replacement overrides on top of the `[TYPE]` default. Unknown type → ValueError.
fn parse_mask_tokens(overrides: Option<HashMap<String, String>>) -> PyResult<MaskTokens> {
    let mut tokens = MaskTokens::default();
    for (name, token) in overrides.unwrap_or_default() {
        let t = EntityType::parse(&name).ok_or_else(|| PyValueError::new_err(format!(
            "unknown entity type {name:?} in mask_tokens; valid: {}", EntityType::ALL.map(|t| t.as_str()).join(", "))))?;
        tokens.set(t, token);
    }
    Ok(tokens)
}

/// Parse user-supplied type names; None = all types. Unknown or empty → ValueError.
fn parse_types(types: Option<Vec<String>>) -> PyResult<Option<Vec<EntityType>>> {
    let Some(names) = types else { return Ok(None) };
    if names.is_empty() { return Err(PyValueError::new_err("types must not be empty (omit it to mask every type)")); }
    names.iter().map(|n| EntityType::parse(n).ok_or_else(|| PyValueError::new_err(format!(
        "unknown entity type {n:?}; valid: {}", EntityType::ALL.map(|t| t.as_str()).join(", "))))).collect::<PyResult<Vec<_>>>().map(Some)
}

#[pymethods]
impl PyMasker {
    #[new]
    #[pyo3(signature = (dir, ort_dylib=None, threads=1, mask_tokens=None))]
    fn new(dir: String, ort_dylib: Option<String>, threads: usize, mask_tokens: Option<HashMap<String, String>>) -> PyResult<Self> {
        if threads == 0 { return Err(PyValueError::new_err("threads must be >= 1")); }
        let tokens = parse_mask_tokens(mask_tokens)?;   // validate before the expensive model load
        if let Some(p) = ort_dylib { init_ort(&p)?; }   // load-dynamic: libonnxruntime from the `onnxruntime` wheel
        let mut inner = Inner::from_dir(std::path::Path::new(&dir), threads).map_err(err)?;
        inner.set_mask_tokens(tokens);
        Ok(PyMasker { inner })
    }

    /// Current replacement text per type, e.g. {"PHONE": "[PHONE]", ...}.
    fn mask_tokens(&self) -> HashMap<String, String> {
        EntityType::ALL.iter().map(|t| (t.as_str().to_string(), self.inner.mask_tokens().get(*t).to_string())).collect()
    }

    /// [(start, end, type, score)] after the decoder. Inference runs with the GIL released; `&mut self`
    /// serializes calls on one Masker (ORT intra-op threads parallelize inside a call).
    #[pyo3(signature = (text, types=None, backstop=false))]
    fn predict(&mut self, py: Python<'_>, text: String, types: Option<Vec<String>>, backstop: bool) -> PyResult<Vec<(usize, usize, String, f32)>> {
        let keep = parse_types(types)?;
        let inner = &mut self.inner;
        let spans = py.detach(move || inner.predict_opts(&text, keep.as_deref(), backstop)).map_err(err)?;
        Ok(spans.into_iter().map(|s| (s.start, s.end, s.entity.as_str().to_string(), s.score)).collect())
    }

    /// Raw model spans (decoder not applied).
    fn predict_raw(&mut self, py: Python<'_>, text: String) -> PyResult<Vec<(usize, usize, String, f32)>> {
        let inner = &mut self.inner;
        let spans = py.detach(move || inner.predict_raw(&text)).map_err(err)?;
        Ok(spans.into_iter().map(|s| (s.start, s.end, s.entity.as_str().to_string(), s.score)).collect())
    }

    #[pyo3(signature = (text, types=None, backstop=false))]
    fn mask(&mut self, py: Python<'_>, text: String, types: Option<Vec<String>>, backstop: bool) -> PyResult<String> {
        let keep = parse_types(types)?;
        let inner = &mut self.inner;
        py.detach(move || inner.mask_opts(&text, keep.as_deref(), backstop)).map_err(err)
    }
}

#[pymodule]
fn _koredact(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMasker>()?;
    m.add("DECODER_VERSION", crate::DECODER_VERSION)?;
    m.add("ENTITY_TYPES", EntityType::TRAINED.map(|t| t.as_str()).to_vec())?;   // model types; NUM is backstop-only
    m.add("BACKSTOP_NUM_TYPE", EntityType::Num.as_str())?;
    let defaults = MaskTokens::default();
    m.add("DEFAULT_MASK_TOKENS", EntityType::ALL.iter().map(|t| (t.as_str(), defaults.get(*t).to_string())).collect::<HashMap<_, _>>())?;
    Ok(())
}
