# koredact

Korean PII masking runtime: BERT token-classification (ONNX, `infobank-corp/koredact-bert-base-onnx`, exported from `infobank-corp/koredact-bert-base`) + rule decoder, Rust core with Python bindings.

## Install

```sh
uv add koredact          # or: uv pip install koredact
```

Python >= 3.11. `onnxruntime` and `huggingface_hub` are pulled in as dependencies; the model bundle is downloaded from the Hub on first use and cached.

## Usage

```python
from koredact import Masker

m = Masker.from_pretrained()                        # infobank-corp/koredact-bert-base-onnx
m.mask("문의 010-1234-5678 홍길동")                 # '문의 [PHONE] [NAME]'
m.predict(text)                                     # [Span(start, end, type, score)]
m.mask(text, types=["PHONE", "EMAIL"])              # mask only these types
Masker.from_pretrained(mask_tokens={"PHONE": "***", "NAME": ""})   # per-type replacement text (default [TYPE])
```

13 entity types: `NAME PHONE EMAIL RRN FRN BRN CARD ACCOUNT ADDRESS DRIVER_LICENSE PASSPORT URL CODE`
(`koredact.ENTITY_TYPES`). Defaults and versions: `koredact.DEFAULT_MASK_TOKENS`, `koredact.DECODER_VERSION`.

## Development

```sh
cargo test --release --no-default-features --features load-dynamic
uv venv && uv pip install maturin && source .venv/bin/activate && maturin develop --release
.venv/bin/python tests/python/smoke_tiny_bundle.py
```

License: Apache-2.0 for this library. Model weights are CC-BY-SA-4.0 (see the model cards).
