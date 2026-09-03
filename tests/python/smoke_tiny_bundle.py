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
assert koredact.BACKSTOP_NUM_TYPE == "NUM" and "NUM" not in koredact.ENTITY_TYPES and koredact.DEFAULT_MASK_TOKENS["NUM"] == "[NUM]"
bs_text = "문의 a@b.com 010-****-1234 주문번호 12345678 #{이름}님"
bs = m.predict(bs_text, backstop=True)
assert {s.type for s in bs} >= {"EMAIL", "PHONE"}, bs          # regex spans present even with random model weights
assert all(not (s.start < bs_text.index("#{") + 5 and s.end > bs_text.index("#{")) for s in bs), bs   # template variable protected
assert "#{이름}" in m.mask(bs_text, backstop=True) and "a@b.com" not in m.mask(bs_text, backstop=True)
assert {s.type for s in m.predict(bs_text, types=["NUM"], backstop=True)} <= {"NUM"}
assert m.predict(bs_text, types=["NUM"]) == []                  # NUM never comes from the model alone
for bad in (["PHONE", "SSN"], []):
    try:
        m.mask(text, types=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"types={bad} should raise ValueError")
print(f"ok · windows exercised · raw {len(raw)} · decoded {len(spans)}")
