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
assert m.mask_tokens == koredact.DEFAULT_MASK_TOKENS and m.mask_tokens["DRIVER_LICENSE"] == "[DRIVER_LICENSE]"
custom = koredact.Masker.from_pretrained(str(bundle), mask_tokens={"PHONE": "***", "NAME": ""})
assert custom.mask_tokens["PHONE"] == "***" and custom.mask_tokens["EMAIL"] == "[EMAIL]"
if any(s.type == "PHONE" for s in spans):
    assert "***" in custom.mask(text) and "[PHONE]" not in custom.mask(text)
try:
    koredact.Masker.from_pretrained(str(bundle), mask_tokens={"SSN": "x"})
except ValueError:
    pass
else:
    raise AssertionError("unknown mask_tokens type should raise ValueError")
for bad in (["PHONE", "SSN"], []):
    try:
        m.mask(text, types=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"types={bad} should raise ValueError")
print(f"ok · windows exercised · raw {len(raw)} · decoded {len(spans)}")
