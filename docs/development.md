# 개발·검증

[README](../README.md)

## 로컬 빌드와 검사

저장소 루트에서 실행. Rust 검사 시 동적 ONNX Runtime 기능 선택.

```sh
cargo test --release --no-default-features --features load-dynamic
uv venv
source .venv/bin/activate
uv pip install maturin
maturin develop --release
.venv/bin/python tests/python/smoke_tiny_bundle.py
```

Python smoke에서 장치 선택·CPU 대체·배치 순서 등 `runtime_options.py` 검사도 함께 실행.
해당 검사만 별도 실행하는 경우:

```sh
.venv/bin/python tests/python/runtime_options.py
```

## 네이티브 실패 복구 검사

설치된 ORT 라이브러리를 지정한 세션 실패·NaN·무한대의 CPU 재시도 검사:

```sh
ORT_DYLIB_PATH="$(.venv/bin/python -c 'from koredact import _onnxruntime_dylib; print(_onnxruntime_dylib())')" \
  cargo test --locked --no-default-features --features load-dynamic failed_accelerator_run -- --ignored
```

- 구조·의존 방향·릴리스 규칙: [AGENTS.md](../AGENTS.md).
- 공개 데이터 대조 결과 및 실기 검증 범위: [런타임 설계 기록](research/2026-09-23-runtime-options.md).
