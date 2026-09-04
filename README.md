# koredact

[![PyPI](https://img.shields.io/pypi/v/koredact)](https://pypi.org/project/koredact/) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/koredact/) [![wheels](https://github.com/baba9811/koredact/actions/workflows/wheels.yml/badge.svg)](https://github.com/baba9811/koredact/actions/workflows/wheels.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE) [![HF model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-koredact--bert--base--onnx-yellow)](https://huggingface.co/infobank-corp/koredact-bert-base-onnx)

한국어 개인정보 마스킹 런타임. BERT 토큰 분류(ONNX `infobank-corp/koredact-bert-base-onnx`,
`infobank-corp/koredact-bert-base` 에서 내보냄) + 규칙 디코더. Rust 코어에 Python 바인딩.

## 설치

```sh
uv add koredact          # 또는: uv pip install koredact
```

Python 3.11 이상. `onnxruntime` 과 `huggingface_hub` 는 의존성으로 함께 설치됨. 모델 번들은 첫 사용 시
Hub 에서 내려받아 캐시됨.

## 사용법

```python
from koredact import Masker

m = Masker.from_pretrained()                        # infobank-corp/koredact-bert-base-onnx
m.mask("문의 010-1234-5678 홍길동")                 # '문의 [PHONE] [NAME]'
m.predict(text)                                     # [Span(start, end, type, score)]
m.mask(text, types=["PHONE", "EMAIL"])              # 지정한 유형만 마스킹
Masker.from_pretrained(mask_tokens={"PHONE": "***", "NAME": ""})   # 유형별 치환 문자열(기본 [TYPE])
```

13 유형: `NAME PHONE EMAIL RRN FRN BRN CARD ACCOUNT ADDRESS DRIVER_LICENSE PASSPORT URL CODE`
(`koredact.ENTITY_TYPES`). 기본값·버전은 `koredact.DEFAULT_MASK_TOKENS`, `koredact.DECODER_VERSION`.

### 백스톱 (opt-in)

`backstop=True` 는 모델 출력 위에 결정적 정규식 안전망을 덧댐 — 놓침이 과마스킹보다 비싼 운영 마스킹용.

- 모델이 라벨하지 않은 `EMAIL`·`URL`·부분 마스킹된 휴대폰 번호(`010-****-1234`) 추가.
- 남은 7자리 이상 숫자열(구분자 `-`·`.`·공백 허용)은 `NUM` 으로 표시 — 유형 판단 없는 포괄 처리라
  놓친 전화·계좌번호뿐 아니라 주문·송장 번호도 함께 걸림. 출력은 `[NUM]`
  (`mask_tokens={"NUM": ...}` 로 변경). `NUM` 은 모델 단독으로는 나오지 않음.
- 템플릿 변수(`#{name}`·`{{code}}`·`${url}`)는 마스킹하지 않음.

정규식 스팬은 확신 있는 모델 스팬보다 낮은 점수를 받음 — 둘 다 걸리면 모델의 유형이 이김. 기본값
`backstop=False` 는 공개 평가 수치의 근거인 모델 + 디코더 계약 그대로.

```python
m.mask("링크 https://example.test/a 주문 12345678 #{이름}님", backstop=True)
# '링크 [URL] 주문 [NUM] #{이름}님'
```

유형이 다른 스팬이 겹쳐도 텍스트가 노출되지 않음 — 겹친 구간은 점수가 높은 쪽이 가져가고 나머지는
각자의 잔여 구간을 유지함.

## 개발

```sh
cargo test --release --no-default-features --features load-dynamic
uv venv && uv pip install maturin && source .venv/bin/activate && maturin develop --release
.venv/bin/python tests/python/smoke_tiny_bundle.py
```

구조·의존 방향·릴리스 규칙은 [AGENTS.md](https://github.com/baba9811/koredact/blob/main/AGENTS.md).

라이선스: 이 라이브러리는 Apache-2.0. 모델 가중치는 CC-BY-SA-4.0(모델 카드 참조).
