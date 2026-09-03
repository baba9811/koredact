"""koredact — Korean PII masking (BERT NER ONNX + rule decoder), Rust core.

    from koredact import Masker
    m = Masker.from_pretrained()            # infobank-corp/koredact-bert-base
    m.mask("문의 010-1234-5678 홍길동")       # '문의 <PHONE> <NAME>'
    m.predict(text)                         # [Span(start, end, type, score)]
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

from . import _koredact

DEFAULT_REPO = "infobank-corp/koredact-bert-base"
BUNDLE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"]
DECODER_VERSION: str = _koredact.DECODER_VERSION


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
    def __init__(self, bundle_dir: str | os.PathLike, *, threads: int = 1, ort_dylib: str | None = None):
        self._inner = _koredact.Masker(str(bundle_dir), ort_dylib or _onnxruntime_dylib(), threads)

    @classmethod
    def from_pretrained(cls, repo_id: str = DEFAULT_REPO, *, revision: str | None = None, threads: int = 1) -> "Masker":
        """Download (cached) the bundle from the Hub, or accept a local directory path."""
        if Path(repo_id).is_dir():
            return cls(repo_id, threads=threads)
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id, revision=revision, allow_patterns=BUNDLE_FILES)
        return cls(local, threads=threads)

    def predict(self, text: str) -> list[Span]:
        return [Span(*t) for t in self._inner.predict(text)]

    def predict_raw(self, text: str) -> list[Span]:
        """Model output after window merge, before the decoder (for diagnostics)."""
        return [Span(*t) for t in self._inner.predict_raw(text)]

    def mask(self, text: str) -> str:
        return self._inner.mask(text)


__all__ = ["Masker", "Span", "DEFAULT_REPO", "DECODER_VERSION"]
