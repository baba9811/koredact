# koredact

[![PyPI](https://img.shields.io/pypi/v/koredact)](https://pypi.org/project/koredact/) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/koredact/) [![wheels](https://github.com/baba9811/koredact/actions/workflows/wheels.yml/badge.svg)](https://github.com/baba9811/koredact/actions/workflows/wheels.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/baba9811/koredact/blob/main/LICENSE) [![HF model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-koredact--bert--base--onnx-yellow)](https://huggingface.co/infobank-corp/koredact-bert-base-onnx)

한국어 개인정보 마스킹 런타임. BERT 토큰 분류(ONNX)와 규칙 디코더 기반의 Rust 코어·Python 바인딩.

## 설치

```sh
uv add koredact
# 또는: uv pip install koredact
```

- Python 3.11 이상, CPU ONNX Runtime·Hugging Face Hub 의존성 포함.
- 첫 사용 시 모델 번들 자동 다운로드 및 캐시.

## 빠른 시작

```python
from koredact import Masker

masker = Masker.from_pretrained()
text = "문의 010-1234-5678 홍길동"
masker.mask(text)     # '문의 [PHONE] [NAME]'
masker.predict(text)  # [Span(start, end, type, score)]
```

- 이름·전화번호·이메일 등 개인정보 13종 지원.
- 가속기 자동 선택 및 CPU 대체, 배치 워커 수 지정 지원. 가속기 사용 시 별도 런타임 설치 필요.

## 문서

- [상세 사용법](https://github.com/baba9811/koredact/blob/main/docs/usage.md): 유형 필터, 치환 문자열, 백스톱.
- [장치 선택·배치 처리](https://github.com/baba9811/koredact/blob/main/docs/usage.md#장치-선택): GPU/NPU/CoreML, 런타임 설치, 워커·스레드 설정.
- [개발·검증](https://github.com/baba9811/koredact/blob/main/docs/development.md): 로컬 빌드, Rust·Python 검사.
- [런타임 설계·검증 기록](https://github.com/baba9811/koredact/blob/main/docs/research/2026-09-23-runtime-options.md): 공식 근거, 측정 결과, 검증 범위.

## 라이선스

- 라이브러리: [Apache-2.0](https://github.com/baba9811/koredact/blob/main/LICENSE).
- 모델 가중치: [CC-BY-SA-4.0](https://huggingface.co/infobank-corp/koredact-bert-base-onnx).
