"""Smoke: load the tiny fixture bundle (random weights, real WordPiece + ONNX graph) through the wheel — proves
dynamic libonnxruntime loading, tokenizer/labels parsing, windowing and decoder plumbing. Types are random."""
from pathlib import Path

import koredact

bundle = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_bundle"
m = koredact.Masker.from_pretrained(str(bundle))
text = "문의 010-1234-5678 홍길동님 " * 40   # > 400 chars → several windows
spans = m.predict(text)
assert all(0 <= s.start < s.end <= len(text) for s in spans), spans
raw = m.predict_raw(text)
assert len(m.mask(text)) > 0 and koredact.DECODER_VERSION.startswith("v3")
assert len(koredact.ENTITY_TYPES) == 13
assert {s.type for s in m.predict(text, types=["PHONE"])} <= {"PHONE"}
assert m.mask(text, types=koredact.ENTITY_TYPES) == m.mask(text)          # all types == default
if spans:
    keep = spans[0].type
    assert m.predict(text, types=[keep]) == [s for s in spans if s.type == keep]
for bad in (["PHONE", "SSN"], []):
    try:
        m.mask(text, types=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"types={bad} should raise ValueError")
print(f"ok · windows exercised · raw {len(raw)} · decoded {len(spans)}")
