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

### Backstop (opt-in)

`backstop=True` layers a deterministic regex safety net over the model output for production masking,
where a miss costs more than an over-mask:

- adds `EMAIL`, `URL` and partially masked mobile numbers (`010-****-1234`) the model did not label;
- marks any remaining run of 7+ digits (separators `-`, `.`, space allowed) as `NUM`, a catch-all with no
  type judgement: missed phone or account numbers, but also order or tracking numbers. Rendered as `[NUM]`
  (`mask_tokens={"NUM": ...}` to change it). `NUM` never comes from the model alone;
- keeps template variables (`#{name}`, `{{code}}`, `${url}`) unmasked.

Regex spans score below every confident model span, so where both fire the model's type wins. The default
(`backstop=False`) is the pure model + decoder contract used for the published evaluation numbers.

```python
m.mask("링크 https://example.test/a 주문 12345678 #{이름}님", backstop=True)
# '링크 [URL] 주문 [NUM] #{이름}님'
```

Overlapping spans of different types never leave text exposed: the higher-scoring span wins the shared
characters and the other keeps its remainder.

## Development

```sh
cargo test --release --no-default-features --features load-dynamic
uv venv && uv pip install maturin && source .venv/bin/activate && maturin develop --release
.venv/bin/python tests/python/smoke_tiny_bundle.py
```

License: Apache-2.0 for this library. Model weights are CC-BY-SA-4.0 (see the model cards).
