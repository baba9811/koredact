"""koredact — Korean PII masking (BERT NER ONNX + rule decoder), Rust core.

    from koredact import Masker
    m = Masker.from_pretrained()            # infobank-corp/koredact-bert-base-onnx
    m.mask("문의 010-1234-5678 홍길동")       # '문의 [PHONE] [NAME]'
    m.mask(text, types=["PHONE"])           # only PHONE masked, everything else left as-is
    m.mask(text, backstop=True)             # + regex safety net (EMAIL/URL/partial PHONE, long digit runs → [NUM])
    Masker.from_pretrained(mask_tokens={"PHONE": "***", "NAME": ""})   # per-type replacement text
    m.predict(text)                         # [Span(start, end, type, score)]
"""
from __future__ import annotations

import glob
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from . import _koredact

DEFAULT_REPO = "infobank-corp/koredact-bert-base-onnx"   # ONNX repo; the PyTorch weights live under the same name without -onnx
BUNDLE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"]
DECODER_VERSION: str = _koredact.DECODER_VERSION
ENTITY_TYPES: tuple[str, ...] = tuple(_koredact.ENTITY_TYPES)           # the 13 model types
BACKSTOP_NUM_TYPE: str = _koredact.BACKSTOP_NUM_TYPE                     # "NUM": backstop-only catch-all for long digit runs
DEFAULT_MASK_TOKENS: dict[str, str] = dict(_koredact.DEFAULT_MASK_TOKENS)   # {"PHONE": "[PHONE]", ..., "NUM": "[NUM]"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    score: float


def _onnxruntime_dylib() -> str:
    """libonnxruntime shipped inside the `onnxruntime` wheel (capi/)."""
    import onnxruntime
    capi = Path(onnxruntime.__file__).parent / "capi"
    hits = [p for pat in ("libonnxruntime*.dylib", "libonnxruntime.so*", "onnxruntime.dll") for p in glob.glob(str(capi / pat))]
    hits = sorted({h for h in hits if "providers" not in os.path.basename(h) and Path(h).is_file()})
    if not hits:
        raise RuntimeError(f"libonnxruntime not found under {capi}")
    # prefer the unversioned runtime name, then shortest (most generic) name, then lexical — deterministic
    return sorted(hits, key=lambda h: (not os.path.basename(h) in ("libonnxruntime.dylib", "libonnxruntime.so", "onnxruntime.dll"),
                                       len(os.path.basename(h)), h))[0]


class Masker:
    def __init__(self, bundle_dir: str | os.PathLike, *, threads: int = 1, ort_dylib: str | None = None,
                 mask_tokens: Mapping[str, str] | None = None):
        """`mask_tokens` overrides the replacement text per type (default `[TYPE]`); unknown types raise ValueError."""
        self._inner = _koredact.Masker(str(bundle_dir), ort_dylib or _onnxruntime_dylib(), threads,
                                       None if mask_tokens is None else {str(k): str(v) for k, v in mask_tokens.items()})

    @classmethod
    def from_pretrained(cls, repo_id: str = DEFAULT_REPO, *, revision: str | None = None, threads: int = 1,
                        mask_tokens: Mapping[str, str] | None = None) -> "Masker":
        """Download (cached) the bundle from the Hub, or accept a local directory path."""
        if Path(repo_id).is_dir():
            return cls(repo_id, threads=threads, mask_tokens=mask_tokens)
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id, revision=revision, allow_patterns=BUNDLE_FILES)
        return cls(local, threads=threads, mask_tokens=mask_tokens)

    @property
    def mask_tokens(self) -> dict[str, str]:
        return self._inner.mask_tokens()

    def predict(self, text: str, types: Iterable[str] | None = None, *, backstop: bool = False) -> list[Span]:
        """Decoded spans. `backstop=True` adds the regex safety net (EMAIL, URL, partially masked PHONE, and
        digit runs of 7+ digits as NUM) wherever the model found nothing, and keeps template variables
        (`#{…}`, `{{…}}`, `${…}`) unmasked. `types` restricts to those entity types (ENTITY_TYPES plus "NUM");
        unknown names raise ValueError."""
        return [Span(*t) for t in self._inner.predict(text, _types(types), backstop)]

    def predict_raw(self, text: str) -> list[Span]:
        """Model output after window merge, before the decoder (for diagnostics)."""
        return [Span(*t) for t in self._inner.predict_raw(text)]

    def mask(self, text: str, types: Iterable[str] | None = None, *, backstop: bool = False) -> str:
        """Masked text. Same options as `predict`; `types` restricts masking, the rest stays in clear text."""
        return self._inner.mask(text, _types(types), backstop)


def _types(types: Iterable[str] | None) -> list[str] | None:
    return None if types is None else [str(t) for t in types]


__all__ = ["Masker", "Span", "DEFAULT_REPO", "DECODER_VERSION", "ENTITY_TYPES", "BACKSTOP_NUM_TYPE", "DEFAULT_MASK_TOKENS"]
