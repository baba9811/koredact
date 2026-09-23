# 장치 자동 선택과 배치 워커

요구사항: 기존 Python/Rust 호출·디코더·마스킹 옵션을 유지하면서 사용 가능한 가속기를
자동 선택한다. 가속기가 없거나 복구 가능한 초기화/추론 오류가 발생하면 CPU로 다시
실행한다. 원문을 결과로 돌려주는 우회는 허용하지 않는다. CPU도 실패하면 기존 오류를
전달한다. 워커 기본값은 1, 기존 intra-op `threads=1`은 유지한다.

## 공식 근거와 적용

- [ort 실행 제공자](https://github.com/pykeio/ort/blob/main/docs/content/perf/execution-providers.mdx):
  빌드에 포함된 제공자 목록과 실제 세션 등록 성공을 구분한다. 각 후보를 별도 세션으로
  시도하고 실패한 세션 옵션을 다음 후보에 재사용하지 않는다. 설치된 2.0.0-rc.13
  crate 원본을 Cargo.lock SHA256과 대조해 API를 확인했다.
- [CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html):
  CUDA/cuDNN 주 버전과 런타임 패키지를 맞춘다. 정밀도 변화 최소화를 위해 CUDA TF32는
  끈다. CPU/GPU 결과 차이는 실제 모델의 공개 평가 텍스트로 별도 검증한다. provider
  등록만으로 모든 연산이 GPU에서 실행됐다고 주장하지 않는다.
- [스레드 관리](https://onnxruntime.ai/docs/performance/tune-performance/threading.html):
  추론 내부 스레드와 동시 문장 워커는 별개다. 자동 CPU 코어 수 확장을 피하고 각 값을
  명시하도록 한다. 배치 워커별 별도 세션을 사용하며 입력 순서와 오류를 보존한다.
- [CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html):
  macOS 가속은 CoreML을 사용한다. metal/mps는 CoreML GPU 경로의 별칭이며 직접 Metal
  실행 제공자가 아니다. NPU 지정은 가능한 경우 Neural Engine을 선호한다.
- [QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html):
  NPU는 SDK·드라이버·모델 연산 및 정밀도 호환성이 필요하다. 제공자 이름만으로 NPU
  지원을 보장하지 않는다. 필요한 런타임이 없거나 모델을 처리하지 못하면 CPU로 대체한다.
- [DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html):
  순차 실행과 memory pattern 비활성화 조건을 유지한다.
- [OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html):
  GPU/NPU 장치 유형을 구분하고 세션 스레드를 명시한다.

## 구현과 검증 범위

기존 단일 문장 메서드는 유지한다. `device="auto"` 및 CPU/GPU/NPU/플랫폼별 선택을
추가하고 선택된 장치를 노출한다. 별도 `predict_many`/`mask_many`에 `workers`를 둔다.
입력·옵션 오류나 손상된 모델은 성공으로 위장하지 않는다. 원문이나 고객 샘플을 이
저장소에 복사하지 않는다. tiny fixture와 공개 평가 자료만 테스트에 사용한다.

이 서버에서는 ARM64/GB10의 CPU·CUDA 경로를 실측한다. 다른 하드웨어는 제공자 선택과
실패 대체 검사까지만 확인할 수 있으므로 macOS/NPU 실기 검증 완료로 표시하지 않는다.
공유 서버 작업은 CPU 4, RAM 20GiB, swap 0, 가용 RAM 12GiB 중단으로 제한한다.

## 로컬 검증 결과

- Rust 단위 21개와 디코더 벡터 1개 통과. 런타임이 필요한 추가 검사도 별도 실행해 통과.
  세션 실패 및 NaN/±Inf를 실제 tiny ONNX 세션에 주입해 동일 입력 CPU 재시도와
  CPU 오류 전파를 확인했다. 기존 upstream 개발셋 비교 검사는 별도 입력이 필요해 제외했다.
- 빌드한 Python 0.4.0 휠로 기존 smoke와 장치·옵션·배치 순서 검사를 통과했다.
- 고정 모델 `5f4f416a12a2b6bd06f999766cbd06fbacbd2cff`, 공개 회귀/합성 1,624건을
  0.3.0 CPU와 비교했다. 0.4.0 CPU·CUDA 모두 마스킹 바이트 및 스팬 위치/유형 차이 0건.
  부동소수점 score의 비트 동일성이나 모든 입력의 동등성을 주장하지 않는다.
- GB10/ARM64, ONNX Runtime CPU/GPU 1.29.0, 워커 2·세션 스레드 1 기준이다.
  두 번의 단일 문장 추론(predict/mask)을 포함한 전수 대조 시간이 CPU 279.01초,
  CUDA 12.65초였다. 한 번의 관측값이며 일반 성능 보장 수치가 아니다.
- 별도 GPU 환경에서 수동 preload 없이 `auto → cuda`를 확인했다. 실제 모델 16건의
  `mask_many`/`predict_many(workers=2)`도 기존 출력·순서와 일치했다.
- macOS/NPU/ROCm/DirectML/OpenVINO의 실제 하드웨어 실행은 미검증이다. CPU 환경의
  제공자 부재 대체와 선택 로직만 검사했으므로 플랫폼별 모델 지원을 보장하지 않는다.

검증 기록: 로컬 `target/verification/`과 분석 저장소
`bizgo-classifier/data/verify/koredact-device-20260923/`에 로그·입력 해시·출력 해시 보관.
