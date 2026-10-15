# 상세 사용법

[README](../README.md)

## 모델과 마스킹 옵션

기본 모델: `infobank-corp/koredact-bert-base-onnx` (`infobank-corp/koredact-bert-base`의 ONNX 내보내기).

```python
from koredact import Masker

masker = Masker.from_pretrained()
text = "문의 010-1234-5678 홍길동"
masker.mask(text)                            # '문의 [PHONE] [NAME]'
masker.predict(text)                         # [Span(start, end, type, score)]
masker.mask(text, types=["PHONE", "EMAIL"])   # 지정한 유형만 마스킹

custom = Masker.from_pretrained(mask_tokens={"PHONE": "***", "NAME": ""})
custom.mask(text)                            # 유형별 치환 문자열 적용
```

- 모델 유형 13종: `NAME PHONE EMAIL RRN FRN BRN CARD ACCOUNT ADDRESS DRIVER_LICENSE PASSPORT URL CODE`.
- 유형 목록: `koredact.ENTITY_TYPES`. 기본 치환 문자열: `koredact.DEFAULT_MASK_TOKENS` (`[TYPE]`).
- 디코더 버전: `koredact.DECODER_VERSION`. 스팬 위치: 문자 오프셋.

## 백스톱

`backstop=True`: 모델 출력에 결정적 정규식 안전망 추가. 누락 방지를 우선하는 운영 마스킹용 옵션.
기본값 `False`: 공개 평가의 기준인 모델 + 디코더 계약 유지.

- 모델이 라벨하지 않은 `EMAIL`·`URL`·부분 마스킹 휴대폰 번호(`010-****-1234`) 추가.
- 남은 7자리 이상 숫자열(구분자 `-`·`.`·공백 허용)을 `NUM`으로 처리.
  전화·계좌번호 외 주문·송장 번호도 포함하는 포괄 마스킹.
- `NUM`의 기본 출력은 `[NUM]`, `mask_tokens={"NUM": ...}`으로 변경 가능.
  모델 단독 출력에는 없는 백스톱 전용 유형.
- 템플릿 변수(`#{name}`·`{{code}}`·`${url}`) 보호.
- 확신 있는 모델 스팬보다 낮은 정규식 스팬 점수 적용. 중복 시 모델 유형 우선.
- 유형이 다른 스팬의 겹침 구간은 높은 점수 우선, 나머지 잔여 구간의 마스킹 유지.

```python
masker.mask("링크 https://example.test/a 주문 12345678 #{이름}님", backstop=True)
# '링크 [URL] 주문 [NUM] #{이름}님'
```

## 장치 선택

```python
masker = Masker.from_pretrained(device="auto", threads=1)
print(masker.device, masker.fallback_reason)
```

| `device` | 선택 범위 |
| --- | --- |
| `auto` (기본값) | 사용 가능한 GPU → NPU → CoreML → CPU 순서 |
| `cpu` | CPU 고정 |
| `gpu` | CUDA → ROCm → DirectML → OpenVINO GPU → CoreML GPU → CPU |
| `npu` | QNN → OpenVINO NPU → DirectML NPU → CoreML Neural Engine → CPU |
| `cuda`, `rocm`, `directml`, `openvino_gpu` | 지정한 GPU 제공자 |
| `qnn`, `openvino_npu`, `directml_npu` | 지정한 NPU 제공자 |
| `coreml` | CoreML |
| `metal`, `mps`, `coreml_gpu` | CoreML CPU+GPU 경로 |
| `coreml_npu` | CoreML CPU+Neural Engine 경로 |

- `metal`/`mps`: 직접 Metal 실행 제공자가 아닌 CoreML 별칭.
- 가속기 등록·모델 초기화·복구 가능한 추론 오류 발생 시 동일 입력의 CPU 재시도.
- CPU 실패 또는 입력·옵션·모델 자체 오류 시 오류 전달. 원문의 성공 결과 반환 금지.
- `device`: 단일 문장 세션에 등록된 제공자. 일부 연산의 CPU 실행 가능.
- 배치 워커: 각 세션에서 독립적인 CPU 대체 가능. 입력별 처리 세션은 호출마다 다를 수 있음.
  가속기 세션과 CPU 대체 세션이 섞이면 점수 등 결과의 미세 차이 가능. CPU 전용 실행 결과는 동일.
- CPU 실행 고정: `device="cpu"`. 기존 인자 생략 호출 유지.

### 가속기 런타임 설치

- 기본 설치에 CPU ONNX Runtime 포함. GPU/NPU 사용 시 해당 제공자가 포함된 런타임과 호환 드라이버·SDK 필요.
- 별도 런타임 라이브러리 지정: `Masker.from_pretrained(ort_dylib="/path/to/libonnxruntime.so")`.
- CPU/GPU Python 배포판의 `onnxruntime` 모듈 공유. 동일 환경 내 중복 설치 금지.
- CUDA/cuDNN 버전 일치 필요. 필요 시 `onnxruntime.preload_dlls()`로 라이브러리 선적재.
- NPU 사용 시 모델 연산·입력 형태·정밀도 조건 충족 필요. 자동 양자화 미지원.

CUDA 13 환경용 별도 설치 예시:

```sh
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python 'onnxruntime-gpu[cuda,cudnn]==1.29.0' 'huggingface_hub>=0.30'
uv pip install --python .venv-gpu/bin/python --no-deps 'koredact==0.4.0'
```

GPU 배포판의 패키지 이름은 `onnxruntime-gpu`, 모듈 이름은 `onnxruntime`.
`uv sync`나 일반 재설치로 기본 의존성을 다시 해결하면 CPU 런타임이 추가될 수 있으므로 별도 환경 관리 필요.

## 배치 처리와 스레드

```python
masked = masker.mask_many(["문의 010-1234-5678", "주문 12345678"], workers=2, backstop=True)
spans = masker.predict_many(["홍길동", "a@example.test"], workers=2)
```

| 옵션 | 의미 | 기본값 |
| --- | --- | --- |
| `threads` | 한 세션의 intra-op 스레드 수 | `1` |
| `workers` | 배치 동시 처리 수 | `1` |

- 입력 순서대로 결과 반환. 기존 `types`, `backstop`, `mask_tokens` 적용.
- 오류 발생 시 가장 앞선 실패 입력의 오류 반환.
- 입력 수 이하로 워커 수 제한. 추가 세션·스레드 생성 실패 시 적은 워커로 처리.
- 추가 세션: 첫 배치 호출에서 생성 후 해당 `Masker` 수명 동안 재사용. 이후 호출은 부족한 수만 추가 적재.
- 메모리: 지금까지 사용한 최대 `workers`(입력 수 이하) × 모델 1개 분량. 워커 수를 줄여도 유지, `Masker` 해제 시 반환.
- 각 워커가 다음 미처리 입력을 차례로 처리. 길이가 고르지 않은 입력도 워커 간 부하 분산.
- `workers × threads`: CPU 할당량(컨테이너 제한 포함) 이하로 지정. 초과 시 세션 간 코어 경합으로 지연 증가.
- `workers=1`부터 자원 사용량 측정 후 증가.

설계 근거·측정 결과: [배치 세션 기록](research/2026-10-15-batch-session-pool.md).

## 마진 게이트 캐스케이드

빠른 모델(양자화 번들 등) 우선 실행, 확신이 낮은 창만 정밀 모델로 재실행하는 opt-in 옵션.
기본 대체 모델·기본 마진 없음. 미지정 시 기존 동작.

```python
from koredact import DEFAULT_REPO, Masker

masker = Masker.from_pretrained(
    "/models/koredact-w8a8",       # 빠른 모델 번들: 로컬 경로 또는 Hub 저장소
    fallback_repo=DEFAULT_REPO,    # 정밀 모델 번들(fp32)
    fallback_margin=2.5,           # 재실행 기준 로짓 차
)
masker.mask("문의 010-1234-5678 홍길동")
windows, rerun = masker.fallback_stats()   # 처리 창 수, 정밀 모델 재실행 창 수
```

- 판정: 400자 창마다 빠른 모델 로짓에서 비특수 토큰별 top-1·top-2 차 계산. 최솟값이 `fallback_margin` 미만이면
  같은 창을 정밀 모델로 재실행, 그 창 전체에 정밀 모델 로짓 사용. 최솟값이 `fallback_margin`과 같거나 크면 빠른 모델 유지.
- 로컬 번들: `Masker(dir, fallback_dir=..., fallback_margin=...)`. 한쪽만 지정, 0 이하·NaN·무한대 마진은 `ValueError`.
  Hub 대체 번들은 `fallback_repo`·`fallback_revision`, 기본 번들과 같은 파일만 다운로드.
- `from_pretrained`: 대체 옵션을 다운로드 전에 검증. `fallback_repo` 없는 `fallback_revision`도 `ValueError`.
- 대체 번들 조건: `config.json` 라벨 맵과 `tokenizer.json`·`tokenizer_config.json` 바이트 일치 필수, 불일치 시 생성 오류.
  ONNX 그래프와 라벨 외 `config.json` 필드는 차이 허용.
- 정밀 모델 세션: 같은 `threads`·`device`, 같은 CPU 재시도 규칙. `device`·`fallback_reason`은 빠른 모델 세션 기준.
- 메모리: 두 모델 동시 적재. 배치 워커마다 두 세션, 최대 `workers` × (빠른 모델 + 정밀 모델).
- 모니터링: `fallback_stats()`는 생성 이후 누적 `(처리 창 수, 재실행 창 수)`, 배치 워커 합산. 재실행 비율 상승 시
  입력 분포 변화 신호, 속도 이득 감소.
- 재실행하지 않은 창의 점수(`score`)는 빠른 모델 값. 정밀 모델과의 동일성 기준은 스팬 위치·유형.
- 마진 선택: 배포 플랫폼·대표 데이터에서 두 모델의 최대 절대 로짓 차 ε 측정 후 `fallback_margin > 2ε`, 여유 포함.
  측정 ε는 미관측 입력의 상한 아님.

근거·측정 결과: [캐스케이드 기록](research/2026-10-15-margin-cascade.md).

## Rust 사용

- 기존 `Masker::from_dir(dir, threads)` 호출 유지.
- 장치 지정: `Masker::from_dir_with_device(dir, threads, "cpu")`.
- 캐스케이드: `Masker::from_dir_with_device(fast, threads, "cpu")?.with_fallback(exact, 2.5)?`, 통계 `fallback_stats()`.
- 가속기 등록: `load-dynamic` 빌드에서 지원. 기본 정적 CPU 빌드에서는 CPU 대체.

공식 근거·실측 결과·하드웨어별 검증 한계: [런타임 설계 기록](research/2026-09-23-runtime-options.md).
