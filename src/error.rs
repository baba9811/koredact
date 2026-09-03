#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")] Io(#[from] std::io::Error),
    #[error("json: {0}")] Json(#[from] serde_json::Error),
    #[error("tokenizer: {0}")] Tokenizer(String),
    #[error("onnx runtime: {0}")] Ort(#[from] ort::Error),
    #[error("model bundle: {0}")] Bundle(String),
}
