# P1 남은 구현 계획

소스 통합 갱신: 별도 Legacy dependency와 legacy/ 폴더를 제거했다.
Legacy route는 src/legacy.rs에서 root src의 native 구현을 연결한다.
제품 Diffusion·공통 Trainer 경로를 유지하고, 구형 Context 사본과 중복 테스트를
정리했다. 보정/통합 매핑은 src/native_provenance/에 보존한다.

갱신: 2026-09-10. 완료 사실은 [현황](P1_STATUS.md),
연산별 제한은 [지원표](P1_LEGACY_OPERATIONS.md)에 기록한다.

## 유지할 구조

현재 Context Diffusion/U-Net과 기존 Trainer를 공통 구현으로 사용한다.
명시적 ctx를 전달하며 생성 시에만 실행 경로를 선택한다.

```rust
let ctx = ExecutionContext::new(); // P1
let ctx = ExecutionContext::builder()
    .route(ExecutionRoute::Legacy)
    .build()?;
```

Legacy에는 legacyBenchmark feature가 필요하다. 미포함 시 DependencyUnavailable,
미지원 요청에는 UnsupportedCapability를 반환한다. P1으로 자동 대체하지 않는다.
모델·Trainer·optimizer·loader·observer에 경로 분기나 native handle을 넣지 않는다.
기존 API·CPU backend·학습 서비스를 재사용하고 필요한 adapter만 보완한다.
원본 실행을 Context VJP로 대체하지 않는다. 원본 보정은 승인 범위와
CORRECTIONS.json에 근거하며 레거시·비교 도구를 보존한다.

## 남은 연산

| 항목 | 남은 작업 |
|---|---|
| 빈 축 broadcasting | 공용 shape 계산의 0과 1 축 처리·gradient 계약 정리. Sub는 shape가 다른 빈 입력을 오류로 차단 |
| Shape 연산 | 추가 Matmul batch/vector broadcasting·singleton batch shape |
| 손실 함수 | mean 외 reduction, 사용자 지정 Huber delta, 고차원 categorical reduction |
| Global Matmax | scalar 최대값과 실제 argmax index 반환 |
| 수치 차이 | ApproxCos 미분 다항식, MAE 오차 0 지점, BCE·CE clipping 차이, 음수 Log 출력(P1 -Inf/native NaN) 계약 정리 |

TopK·Matmax tracked differentiation은 P1에서도 미지원이다.
이를 Legacy 연결 누락으로 간주하지 않으며 지원 범위는 별도로 정한다.
Tanh/Softmax backward 입력 계약은 해결됐다. Tanh의 기존 exp 순전파에는
큰 절댓값에서 overflow 가능성이 남아 있다.

Sigmoid 연결 및 출력/gradient/no-grad 비교 검증은 완료했다.
Div·Sum 수정과 공통 Diffusion·DataLoader·Trainer·Adam 3 epoch 학습 E2E 검증도 완료했다.
동일 noise의 sampling 각 step·최종 결과 비교도 완료했다.
Abs·Log·Sqrt native backward도 완료했다. Abs의 0 지점 gradient는 0이며,
Log/Sqrt의 기존 순전파 정의역 동작은 유지한다.
Sub의 nonempty trailing-axis broadcasting 및 원래 입력 shape로의 gradient 합산도
완료했다. Native 대입 경로는 같은 forward를 사용한다.
단일 입력 Concat은 native 입력 개수 제한을 완화하고 기존 forward/backward를
그대로 연결했다. 일반 Transpose 순열은 native 축 교환을 합성하며, rank 0/1은
native Reshape로 처리한다. 다음 우선순위: **추가 Matmul broadcasting 계약**.
나머지 연산도 지원 대상으로 유지한다. 구체적인 차이를 확인하며 진행하고,
연산 연결만으로 P1 전체 완료를 선언하지 않는다.

## 실행 경로 검증과 완료 기준

제품 직접 학습 경로는 `tests/diffusion_routes.rs`로 별도 검증했다.
동일 제품 Diffusion·Adam·공통 UnsupervisedTrainer에 ctx route만 변경하며
원본 fixture와 모델 래퍼를 사용하지 않는다. 3 epoch loss·전체 가중치 갱신·
정리 비교가 통과했다. 원본 재생 테스트의 예측·gradient 비교와는 별도 증거다.

원본 baseline, Legacy adapter, P1을 구분해 비교한다.
같은 초기 가중치·배치·실제 timestep/noise를 사용하며 seed만 맞추지 않는다.
이름·shape·공유 관계 기반 version-2 fixture를 재사용한다.

기준: 채널 1의 8×8 이미지, dim=8, multipliers=[1,2], groups=4,
down/up attention=[false,false]와 middle attention, linear T=10,
beta=1e-4..0.02, 값 0.5인 sample 두 개, batch=2, shuffle=false.
Adam(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8),
UnsupervisedTrainer 3 epoch, tolerance=1e-10을 유지한다.

- 완료: 공통 U-Net 예측, loss, 모든 parameter gradient와 Adam 첫 step·3 epoch 갱신을 원본 fixture와 비교. 개별 블록 검사는 기존 테스트 범위를 유지한다.
- 완료: 동일 Context 모델·Trainer의 native/P1 선택과 원본 graph 생성 여부 확인.
- 완료: 동일 noise의 제품 sampling 10단계 prediction·image·최종 결과 비교. 반복 실행과 중간 shape 오류의 임시 tensor/graph 정리, 입력 보존, 오류 후 추적 복원도 검증.
- 반복 배치, 성공/오류/no-grad, 반환 handle 수명, graph/gradient 정리 검증.
- 외부 provider·feature 제외·공유 parameter·observer 실패·checkpoint 경계 검사.
- Legacy custom op, detach, explicit seed, capture, 부분 graph 정리 등 미지원
  실행 기능의 범위와 실패 계약 확인. 혼합 provider를 묵시적으로 허용하지 않음.

수치 기본 기준은 abs 또는 rel 1e-3이며 NaN/Inf는 실패다.
불일치는 최초 분기 지점을 기록하고 허용치 확대·ignore로 감추지 않는다.
전체 MLP는 Sigmoid 수정 이후에도 출력 표현·모델 계약을 맞춰 별도 검증한다.
완전 resume와 optimizer snapshot은 P2로 유지한다.

수치 검증 후 release benchmark를 실행한다. 초기화/변환 비용, 연산,
forward/backward, Trainer+Adam, sampling을 구분해 median/p95, throughput,
환경·소스 metadata를 남긴다. live handle과 allocator 메모리를 혼동하지 않는다.
과거 benchmark를 수정된 기준의 결과로 인용하지 않는다.

## P1 이후: 정적 그래프

현재 eager 동적 그래프를 P1 완료 후 별도 단계로 전환한다.
학습 시작 전 forward/backward의 수명·참조·마지막 사용, 보존할 값,
버퍼 배치/재사용·필요 복사를 분석한다. 이는 텐서 값을 미리 계산하는 단계가
아니라 실행 흐름과 메모리 사용 계획을 만드는 단계다.

반복 배치는 계획과 버퍼를 재사용한다. 입력은 가능한 경우 참조로 연결하고
출력은 배정된 버퍼에 기록한다. 이전 값 보존과 덮어쓰기가 충돌하거나 커널 배치
요구가 있을 때 복사를 계획한다. 외부 입력 수명 보장이 어려우면 최초 복사를 허용한다.

실행 중 shape·alias·참조 유효성과 데이터 의존 분기를 처리한다.
조건 변경 시 호환 계획을 선택하거나 실행 전에 재분석한다.
Trainer에 메모리 관리 책임을 추가하지 않는다. 수치 동등성, 이전 배치 참조 누적,
오류 정리, 메모리·성능을 검증한다.

일반 snapshot 복사는 시각화 feature와 무관하다. 시각화 비활성 조건에서 비용을
측정한다. 시각화 활성 시에만 발생하는 추가 비용은 필수 최적화 대상이 아니다.
