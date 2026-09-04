//! WordPiece tokenizer wrapper — char offsets (Python fast-tokenizer `offset_mapping` coordinates).
use std::path::Path;
use tokenizers::{Encoding, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::error::Error;

pub struct Tok {
    inner: Tokenizer,
}

impl Tok {
    /// `tokenizer.json` + `model_max_length` (from tokenizer_config.json). Truncation mirrors the
    /// transformers pipeline: `truncation=True, max_length=model_max_length`, longest-first, right.
    pub fn load(tokenizer_json: &Path, max_length: usize) -> Result<Tok, Error> {
        let mut inner = Tokenizer::from_file(tokenizer_json).map_err(|e| Error::Tokenizer(e.to_string()))?;
        inner.with_truncation(Some(TruncationParams {
            max_length, strategy: TruncationStrategy::LongestFirst, stride: 0, direction: TruncationDirection::Right,
        })).map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Tok { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Encoding, Error> {
        self.inner.encode_char_offsets(text, true).map_err(|e| Error::Tokenizer(e.to_string()))
    }
}
