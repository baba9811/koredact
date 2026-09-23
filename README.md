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

### 장치 선택과 배치 처리

```python
m = Masker.from_pretrained(device="auto", threads=1)   # 기존 인자 생략 호출도 유지
print(m.device, m.fallback_reason)                   # 선택한 제공자와 대체 사유
masked = m.mask_many(["문의 010-1234-5678", "주문 12345678"], workers=2, backstop=True)
spans = m.predict_many(["홍길동", "a@example.test"], workers=2)
```

`auto`는 사용 가능한 GPU(CUDA/ROCm/DirectML/OpenVINO), NPU(QNN/OpenVINO/DirectML),
CoreML 순으로 세션을 시도하고 마지막에 CPU를 사용한다. `cpu`, `gpu`, `npu` 또는
`cuda`, `rocm`, `directml`, `openvino_gpu`, `qnn`, `openvino_npu`, `directml_npu`,
`coreml`로 범위를 지정할 수 있다. `metal`/`mps`/`coreml_gpu`는 CoreML의 CPU+GPU 경로,
`coreml_npu`는 CPU+Neural Engine 경로다. 직접 Metal 실행 제공자는 아니다.

가속기 등록·모델 초기화·복구 가능한 추론 오류는 CPU로 대체한다. CPU도 실패하거나
입력/옵션/모델 자체가 잘못됐으면 오류를 전달하며 원문을 성공 결과로 돌려주지 않는다.
`device`는 단일 문장 세션에 등록된 제공자이며, 일부 연산의 CPU 실행 가능성까지 배제하지
않는다. 배치 워커도 각 세션에서 독립적으로 대체할 수 있다.

기본 설치는 CPU ONNX Runtime을 포함한다. GPU/NPU는 해당 제공자를 포함한 ONNX Runtime과
호환 드라이버·SDK가 있어야 한다. `ort_dylib`로 별도 런타임 라이브러리를 지정할 수도 있다.
CPU/GPU Python 런타임 배포판은 같은 `onnxruntime` 모듈을 쓰므로 같은 환경에 겹쳐 설치하지
않는다. GPU 런타임을 관리하는 환경에서는 기본 의존성의 CPU 런타임 재설치 여부도 확인한다.
CUDA/cuDNN 버전을 맞추고 필요하면 `onnxruntime.preload_dlls()`로 라이브러리를 먼저 적재한다.
NPU는 모델의 연산·입력 형태·정밀도 조건도 만족해야 한다. 모델을 자동 양자화하지 않는다.

CUDA 13 환경의 별도 GPU 설치 예시:

```sh
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python 'onnxruntime-gpu[cuda,cudnn]==1.29.0' 'huggingface_hub>=0.30'
uv pip install --python .venv-gpu/bin/python --no-deps 'koredact==0.4.0'
```

GPU 배포판은 Python 모듈은 같지만 패키지 이름이 다르다. 기본 CPU 의존성을 다시
해결하는 `uv sync`/일반 재설치는 CPU 런타임을 추가할 수 있으므로 위 환경은 별도로 관리한다.

`threads`는 한 세션의 intra-op 스레드 수(기존 기본 1), `workers`는 배치 동시 처리 수
(기본 1)다. 결과는 입력 순서이며 기존 `types`, `backstop`, `mask_tokens`가 그대로 적용된다.
워커 수는 입력 수 이하로 제한하고, 추가 세션이나 스레드를 만들 수 없으면 적은 워커로
처리한다. 워커마다 모델 메모리가 추가되므로 1부터 측정해 늘린다. 추가 세션은 배치 호출
동안만 유지된다. CPU 재현성을 고정하려면 `device="cpu"`를 사용한다.

Rust의 기존 `Masker::from_dir(dir, threads)`도 유지한다. 장치를 지정하려면
`Masker::from_dir_with_device(dir, threads, "cpu")`를 사용한다. 가속기 등록은
`load-dynamic` 빌드에서 지원하고 기본 정적 CPU 빌드는 CPU로 대체한다.

공식 근거와 검증 범위: [런타임 설계 기록](docs/research/2026-09-23-runtime-options.md).

### 백스톱 (opt-in)

`backstop=True` 는 모델 출력 위에 결정적 정규식 안전망을 덧댐(놓침이 과마스킹보다 비싼 운영 마스킹용).

- 모델이 라벨하지 않은 `EMAIL`·`URL`·부분 마스킹된 휴대폰 번호(`010-****-1234`) 추가.
- 남은 7자리 이상 숫자열(구분자 `-`·`.`·공백 허용)은 `NUM` 으로 표시: 유형 판단 없는 포괄 처리라
  놓친 전화·계좌번호뿐 아니라 주문·송장 번호도 함께 걸림. 출력은 `[NUM]`
  (`mask_tokens={"NUM": ...}` 로 변경). `NUM` 은 모델 단독으로는 나오지 않음.
- 템플릿 변수(`#{name}`·`{{code}}`·`${url}`)는 마스킹하지 않음.

정규식 스팬은 확신 있는 모델 스팬보다 낮은 점수를 받음. 둘 다 걸리면 모델의 유형이 이김. 기본값
`backstop=False` 는 공개 평가 수치의 근거인 모델 + 디코더 계약 그대로.

```python
m.mask("링크 https://example.test/a 주문 12345678 #{이름}님", backstop=True)
# '링크 [URL] 주문 [NUM] #{이름}님'
```

유형이 다른 스팬이 겹쳐도 텍스트가 노출되지 않음: 겹친 구간은 점수가 높은 쪽이 가져가고 나머지는
각자의 잔여 구간을 유지함.

## 개발

```sh
cargo test --release --no-default-features --features load-dynamic
uv venv && uv pip install maturin && source .venv/bin/activate && maturin develop --release
.venv/bin/python tests/python/smoke_tiny_bundle.py
.venv/bin/python tests/python/runtime_options.py
```

Python smoke는 런타임 옵션 검사도 호출하므로 기존 CI 진입점에서 함께 실행된다.
NaN/무한대 등 네이티브 실패 복구 검사는 설치된 ORT 라이브러리를 지정해 실행한다.

```sh
ORT_DYLIB_PATH="$(.venv/bin/python -c 'from koredact import _onnxruntime_dylib; print(_onnxruntime_dylib())')" \
  cargo test --locked --no-default-features --features load-dynamic failed_accelerator_run -- --ignored
```

구조·의존 방향·릴리스 규칙은 [AGENTS.md](https://github.com/baba9811/koredact/blob/main/AGENTS.md).

라이선스: 이 라이브러리는 Apache-2.0. 모델 가중치는 CC-BY-SA-4.0(모델 카드 참조).
