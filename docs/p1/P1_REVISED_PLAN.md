# P1 수정 계획 — 기준 DDPM 모델의 실행 경로 전환

작성일: 2026-09-07. 갱신: 2026-09-08. 현재 Context Diffusion과 명시적 ctx 구조를
공통 구현 기준으로 확정한다. 기본 경로는 P1이며 Legacy만 필요할 때 명시한다.
후속 사용자 결정: Conv2D 입력 gradient 오류는 원본 레거시에서 직접 수정하고
수정된 원본을 비교 기준으로 사용한다. 아래 원본 수식 불변 규칙의 명시적 예외이며,
`legacy/CORRECTIONS.json`에 기존/수정 해시와 사유를 기록한다. 별도 보정 adapter는 만들지 않는다.
이 문서는 최초 작성 시 설계안이다. 이후 구현·검증의 현재 상태는
`P1_HANDOFF.md`와 `P1_STATUS.md`를 따른다.

## 1. 목표와 이전 계획의 수정

후속 갱신: 사용자 승인으로 Tanh/Softmax 입력 재계산 역전파를 구현하고 tracked
Legacy route를 연결했다. 아래 Softmax 해결 대기 조건은 해소됐으며, 다음 확인은
공통 Context Diffusion의 Legacy 경로 E2E이다. Sigmoid 사용자 수정과 함께
CORRECTIONS.json에 원본 보정 provenance를 기록했다.

2026-09-08 연산 일괄 연결 이후 남은 작업과 원본 계약 제약은
`P1_LEGACY_OPERATIONS.md`에 기록한다. 원본 호출이 가능한 공간 연산·저장 텐서·
mean loss·다중 출력 연결은 구현했으며, 원본 수식/역전파 계약이 다른 항목을
자동 보정하거나 P1으로 우회하지 않는다. tracked Softmax 해결 전에는 공통
Context Diffusion의 Legacy 학습 경로를 완료로 판정하지 않는다.

현재 `src/nn/diffusion.rs`의 Context 버전 Diffusion/U-Net을 구현 기준으로 삼는다.
사용자가 ctx를 생성하고 같은 모델과 Trainer에 전달하는 구조를 유지한다.
기존 `ExecutionContext::new()`는 P1 경로로 동작하며, Legacy가 필요할 때만
ctx 생성 시 route를 명시한다. 모델·Trainer 내부에는 경로 분기를 넣지 않는다.
레거시 원본 Diffusion은 수정된 Conv2D를 포함한 수치 비교 기준으로 보존하며,
사용자용 모델 구현을 원본 레거시 구조로 되돌리거나 두 벌로 새로 작성하지 않는다.

근거 문서는 `P1_STATUS.md`, 기존 첨부 계획 `message(6).txt`와 `message(7).txt`,
대화에서 수립한 P1 구현 계획이다. 첨부 문서는 과거 설계 자료이며 현재 지시가 아니다.
이 문서는 최신 사용자 요구와 충돌하는 다음 항목을 대체한다.

- 실제 실행 엔진은 Context 하나뿐이라는 제한을 재검토한다.
- 레거시를 비교 전용으로만 격리하는 것을 완료 목표로 삼지 않는다.
- P1용 모델을 별도로 재작성하여 비교하는 것만으로 모델 호환성을 입증하지 않는다.
- 현재 Context 모델의 계산 구조를 재사용하고 실행 경계 아래에서 route를 선택한다.

유지하는 요구: 기존 backend CPU 재사용, 구현 독립 계약, 구조화된 오류,
공통 학습 서비스, 명시적 context, 안정적인 parameter 식별, 수명·정리 보장,
레거시 보존, 수치 검증 이후 성능 측정. 삭제 결정은 사용자에게 남긴다.
optimizer snapshot과 완전 resume는 기존대로 P2다.

## 2. 정확한 기준 모델과 실행 조건

구현 기준: `src/nn/diffusion.rs`의 `Diffusion`, `Unet`, `DiffusionScheduler`.
기존 8×8·dim=8·Adam 3 epoch fixture는 연속적인 수치 검증을 위해 유지한다.
비교 원본: `legacy/src/tests/common/model/diffusion/mod.rs:650`
`diffusion_train_with_trainer`. 모델은 해당 파일의 `Diffusion`이며
`unet.rs`, `embedding.rs`, `encoder.rs`, `scheduler.rs`에 연결된다.

| 항목 | 원본 기준 |
|---|---|
| 이미지 | 채널 1, 8×8 |
| U-Net | dim=8, dim_mults=[1,2], init_dim/out_dim 기본값 |
| GroupNorm | groups=4 |
| attention | down/up 설정 [false,false]; middle attention은 존재 |
| scheduler | linear, T=10, beta_start=1e-4, beta_end=0.02 |
| 입력 | 값 0.5인 [1,8,8] sample 두 개 |
| loader | MemorySource→DatasetBuilder→UnsupervisedStackCollator, batch=2, shuffle=false |
| optimizer | Adam(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8) |
| 학습 | UnsupervisedTrainer, 3 epoch, tolerance=1e-10 |
| 검증 | 유한 loss, 완료 epoch=3, avg_loss/grad_norm/update_ratio/forward_secs/backward_secs |

실제 코드의 Trainer는 `Trainer::verbose().unsupervised()`이다.
인접 주석의 silent 설명보다 실행 코드를 기준으로 한다.
시간 지표는 존재·유한성·의미를 비교하며 두 경로의 시간값 일치를 요구하지 않는다.

학습은 batch당 timestep 하나를 선택하고 Gaussian noise를 생성하며,
q_sample→t/T 정규화→U-Net→MSE로 이어진다. 기존 테스트는 학습만 검증한다.
sampling은 P1의 기존 범위로 유지하되, 별도의 동일-noise 역확산 검증으로 추가한다.

현재 4×4·dim=2·SGD fixture는 진단용 보조 테스트다. 기준 DDPM·Adam·Trainer
전체 경로의 대체 증거로 사용하지 않는다.

## 3. 현재 성과의 재사용과 부족한 증거

### 최소 변경 원칙 — 확정 요구

상위 계층 재작성을 전제로 삼지 않는다. 이전 P1에서 이미 구현한 추상화·공개 API·
공통 실행 서비스를 먼저 코드와 테스트로 확인하고, 부족한 경계만 보완한다.
문서의 표현이나 타입 이름이 아니라 실제 의존성과 동작으로 재사용 가능 여부를 판단한다.

변경 우선순위는 다음과 같다.

1. 기존 공개 계약·구현을 그대로 재사용한다.
2. import, 타입 참조, 연산 API 호출의 기계적 치환으로 계약을 충족하면 그것을 선택한다.
3. 치환만으로 해결되지 않는 의미 차이는 공통 API 또는 adapter의 최소 보완으로 처리한다.
4. 그 방법들로도 해결할 수 없는 구체적인 결합이 확인된 부분만 국소적으로 재구성한다.

기계적 변경의 충분성은 forward/backward 추적, parameter 공유·식별, no-grad,
수명·cleanup, 오류 전파, 관찰 이벤트 의미가 유지되는지로 검증한다.
컴파일만 통과하는 이름 치환으로 완료 처리하지 않는다. 반대로 이 검증이 통과하면
새로운 상위 추상화나 Trainer·모델·레이어 구현을 추가할 이유로 삼지 않는다.

구현 전 의존성 감사 결과에 대상별 현재 API, 실제 내부 결합, 재사용 가능한 구현,
최소 치환안, 필요한 adapter 보완, 검증 항목을 기록한다. 재구성이 필요한 경우에는
기계적 치환이나 adapter 보완으로 해결되지 않는 구체적 이유와 변경 범위를 남긴다.
기존 P1 코드를 새 이름으로 복제하거나 동등한 상위 계층을 두 벌로 구현하지 않는다.
원본 레거시 비교 스냅샷 보존은 이 중복 구현 금지와 별개다.

`P1_STATUS.md`에 기록된 119개 테스트와 독립 provider 테스트는 기존 성과로 유지한다.
이는 이번 재계획에서 새로 실행한 결과가 아니며, 구현 시작 시 다시 기준을 고정한다.

재사용: contracts, context 수명/graph cleanup, backend CPU, 공통 Trainer,
loader, RNG 분리, capture, checkpoint 코드와 현재 DDPM 구현 및 비교 도구.
현 Context DDPM을 공통 실행 모델로 재사용하고 원본과의 수치 비교를 유지한다.

부족한 증거:

- 레거시 모델과 Context 모델이 별개이며 단일 진입점에서 선택할 수 없다.
- 작은 reference provider 교체는 실제 레거시 DDPM의 호환성을 입증하지 않는다.
- 현재 비교는 기준 모델의 Adam 3 epoch E2E 및 동일-noise sampling을 다루지 않는다.
- 저장소 위치와 feature 격리는 실행 경로 교체 계약이 아니다.

## 4. 전환의 의미와 경로의 범위

우선 목표는 같은 binary에서 시작 옵션 하나로 경로를 고르는 것이다.
아래 이름은 제안이며 공개 API 확정안이 아니다.

```text
diffusion_train --fixture reference                 # 기본 P1
diffusion_train --execution legacy --fixture reference
diffusion_compare --fixture reference
```

두 구현이 포함된 빌드에서는 경로 선택 때문에 재컴파일하거나 모델 소스를 수정하지 않는다.
필요 구현이 제외된 빌드에서도 요청 API는 남고, 선택 시 DependencyUnavailable을 반환한다.
feature는 사용 가능한 구현 집합을 결정하고 실행 옵션은 실제 경로를 결정한다.
`legacyBenchmark`를 전환용 feature로 재사용할지 별도 feature를 둘지는 이름/용도 감사 후 결정한다.

실행 중 살아 있는 graph·handle을 다른 엔진으로 옮기는 hot swap은 우선 목표에 포함하지
않는 안을 제안한다. 경로 재선택은 새 실행 세션에서 수행한다. 가중치 전송은 가능하게 하되
optimizer state까지 유지하는 중간 학습 resume로 표현하지 않는다. 이 범위는 설계 단계에서 확인한다.

구분해야 할 두 종류의 교체:

1. **연산 구현 교체:** Context의 저장소·autograd 아래 레거시 forward/VJP를 연결.
   연산 provider 교체 검증에는 유효하지만 원래 레거시 graph 경로 실행은 아니다.
2. **실행 경로 교체:** 레거시 Variable/Function/graph/backward 경로와 Context 경로를
   현재 Context 모델/학습 경계 아래 각각 연결. 이번 전환 목표다.

보고서에는 사용한 forward·backward·storage 구현을 명시한다. 레거시 이름으로
P1 backward를 실행하는 혼합 경로를 원본 경로와 동등하게 취급하지 않는다.
경로별 호출 계수/추적 테스트로 실제 선택을 검증한다.

## 5. 확정한 모델 구조와 Legacy 전달 경계

현재 Context Diffusion/U-Net의 모델 계산 코드를 공통으로 사용한다. 기존 A/B/C
통합·분리 선택 논의는 이 결정으로 대체한다. 명시적 ctx 생성과 모델·Trainer에
ctx를 전달하는 사용법을 유지하며, 일반 사용자는 route를 지정할 필요가 없다.

다음은 제안 API이며 아직 구현된 기능이 아니다. 정확한 타입·메서드 이름은 구현 시 정한다.

```rust
let ctx = ExecutionContext::new(); // 기본 P1 경로, 현재 사용법 유지
let ctx = ExecutionContext::builder()
    .route(ExecutionRoute::Legacy) // 레거시가 필요할 때만 명시
    .build()?;
// 이후 같은 Context Diffusion과 Trainer에 &ctx를 전달한다.
```

Legacy adapter는 공개 요청을 원본 Variable/연산/graph/backward에 전달한다.
레거시 내부 계산 순서·수식·역전파 알고리즘을 재구성하거나 Context VJP로 대체하지
않는다. adapter는 handle 대응, 오류 변환, 세션 소유·cleanup 경계만 연결한다.
현재 모델에서 나온 연산 요청을 전달하는 것이며, 레거시 Diffusion/Trainer 전체를
대신 호출하여 모델 소스 공유가 된 것으로 처리하지 않는다. 승인된 Conv2D 보정은 유지한다.

한 실행의 handle은 선택한 route에 속하며 연산마다 두 경로 사이로 변환·복사하지
않는다. 지원하지 않는 요청은 명시적 오류로 반환한다. Legacy 구현이 빌드에서
제외됐으면 DependencyUnavailable을 반환하고 P1으로 조용히 대체하지 않는다.
현재 facade의 직접 backward/cleanup 책임은 필요한 최소 실행 경계로 분리하되,
상위 모델·Trainer에는 경로별 분기나 원본 내부 타입 접근을 추가하지 않는다.

다음 검증은 현재 Context TimeEmbedding + ResidualBlock의 연산 흐름을 양쪽
route에서 실행하는 것이다. 출력·gradient·parameter 갱신·no-grad·오류 정리와
실제 원본 호출 경로를 검증한 후 전체 Context Diffusion으로 확장한다.
명명 parameter 매핑과 version-2 재생 fixture를 재사용한다. 정적 그래프 전환은
기존 결정대로 P1 완료 후 별도 단계이며 이 route 선택 작업과 혼동하지 않는다.

## 6. 어떤 모델 배치에서도 지켜야 할 계약

### 상위 계층의 구현 독립성 — 확정 요구

의존 방향은 다음과 같다. 화살표는 호출/사용 방향이다.

```text
Trainer · 모델 · 레이어 · optimizer · loader · 진단/observer
                         ↓ 공개 API만 사용
       공개 추상화 계층 (계약, handle, 학습 scope, 실행 서비스)
                         ↓ 주입된 구현
        레거시 연산/실행 adapter 또는 Context 연산/실행 adapter
```

이미 구현된 Trainer는 기존 공개 API 사용을 먼저 확인하고 필요한 최소 치환·보완 이후,
레거시↔Context 전환이나 새로운 규격 준수 구현 추가 때문에 수정하지 않는다.
현재 Context에 결합된 Trainer를 그대로 두고 레거시 전용 Trainer를 추가하는 것은
이 요구를 충족하지 않는다. 기존 레거시 Trainer 원본은 비교용으로 계속 보존하되,
사용자에게 제공하는 경로 전환 기능은 동일한 Trainer 구현을 사용한다.

이 요구는 모델 통합/분리 선택과 무관하게 확정이다. 모델을 분리하는 안을 선택하더라도
양쪽 모델은 동일한 공개 학습 계약을 제공하며 Trainer를 분리하지 않는다.

공개 API는 tensor/parameter 접근, forward 결과, backward 요청, no-grad,
학습 scope 생성·정리, gradient 조회·갱신, capability 확인과 진단 snapshot을
구체 구현을 노출하지 않고 표현해야 한다. 기존 공개 ExecutionContext 이름은 유지할 수
있지만, 그것이 특정 Context 엔진 구현에만 결합된 타입이면 경계가 완성된 것이 아니다.
내부 타입을 단순히 pub으로 노출하는 것도 공개 추상화로 인정하지 않는다.

상위 계층에는 legacy/context별 분기, 구체 provider downcast, graph/registry 직접 접근,
구현별 handle 변환을 넣지 않는다. 경로 선택은 애플리케이션의 생성·주입 지점에서 끝내고,
의미 차이와 수명·오류 변환은 adapter 아래에서 처리한다. 미지원 기능도 같은 공개 오류로
전달하여 Trainer가 내부 구현을 알아야 오류를 처리하는 상황을 만들지 않는다.

검증은 외부 integration test에서 내부 모듈을 import하지 않는 사용자 Trainer/모델/
optimizer/loader/observer를 작성하고, 같은 소스·같은 Trainer 타입으로 두 경로를
실행한다. built-in Trainer에도 동일 검증을 적용한다. 경로 추가/교체 변경의 diff에
Trainer 알고리즘 수정이 필요하다면 추상화 누수로 판정한다. 모든 feature 조합에서
상위 모듈의 내부 구현 import와 경로 분기를 검사하는 의존성 검증도 추가한다.

### 세부 계약

- 모델은 필요한 연산·shape·gradient 의미만 요청한다. registry, 구체 NodeId,
  전역 storage, CPU kernel을 직접 접근하지 않는다.
- 모델에 경로별 분기를 반복 삽입하지 않는다. 차이는 adapter와 생성 경계에서 처리한다.
- 레거시의 전역 graph는 adapter 내부에서 세션 단위로 관리한다. 동시/중첩 실행을
  보장할 수 없다면 명시적으로 거부하고, 다른 세션의 graph를 clear하지 않는다.
- tensor/parameter 소유 경로를 검증한다. foreign handle을 암묵 복사하거나 허용하지 않는다.
- 경로 간 값 이동은 명시적 owned buffer/weight snapshot으로 수행한다.
  변환 비용은 숨기지 않고 경계 비용에 포함한다.
- 경로 내부 ParameterId와 경로 간 parameter 이름을 구분한다. 경로 간 매핑은
  구조적 이름·shape·dtype·공유 관계로 검증하며 Vec 순서 zip만으로 대응하지 않는다.
- 학습 서비스는 검증→scope→forward→backward→진단/clipping/hook→optimizer→cleanup→
  성공 이벤트 의미를 공유한다. legacy Trainer를 통째로 호출하는 wrapper만으로
  공통 Trainer 완성을 주장하지 않는다. 원본 Trainer는 비교 기준으로 별도 실행 가능하다.
- 구현 제거 시 필요한 요청은 DependencyUnavailable, 구현은 있으나 능력이 없으면
  UnsupportedCapability를 반환한다. 원래 오류와 cleanup 오류를 함께 보존한다.
- 선택하지 않은 기능의 구현을 강제하지 않는다. 명시적으로 요청한 attention/gradient/
  capture 기능을 조용히 끄거나 다른 연산으로 대체하지 않는다.
- 외부 작성 저장소·autograd·연산 교체 검증은 유지한다. 엔진 전체를 선택하는 기능이
  하위 모듈 각각의 독립성을 대신하지 않는다.

## 7. 원본 보존과 비교 설계

`legacy/src` 원본 142개와 hash manifest는 보존한다. 이 스냅샷은 수치 기준이며
새 선택 경로에서 호출할 수 있어야 한다. 디렉터리 이동 자체는 이번 목표가 아니다.
기존 원본을 연결할 수 없는 부분은 별도 adapter/호환 코드로 조정하고,
원본과 달라진 생성·호출·정리 동작을 목록화한다. 원본의 수식을 수정하지 않는다.

비교는 세 대상을 구분한다: 원본 baseline, 선택 가능한 legacy 경로, 선택 가능한 Context 경로.
baseline→legacy adapter 비교로 adapter가 의미를 바꾸지 않았음을 먼저 확인한 뒤,
legacy→Context를 비교한다. 기존 P1 모델과 새 adapter가 서로만 일치하는 것으로 충분하지 않다.

공통 fixture에 모델 설정, 이름 기반 초기 가중치, batch 순서, epoch별 timestep,
Gaussian noise, sampling 초기값과 단계별 noise를 저장한다. seed만 맞추지 않는다.
원본 RNG 호출 순서가 다르므로 생성된 값을 기록·재생하는 지점이 필요하다.
그 연결은 원본 수식 변경과 구분하고 fixture 및 adapter provenance에 남긴다.

순차 비교:

1. parameter 이름·shape·공유 관계, q_sample, timestep embedding.
2. residual/attention/down/up 블록과 U-Net 예측.
3. loss와 모든 parameter gradient; backward 미지원은 조용히 제외하지 않는다.
4. 동일 초기 상태의 Adam 첫 step 및 3 epoch 후 모든 가중치·loss.
5. 원본 loader/collator 조건을 거친 공통 Trainer E2E, metric 의미, scope 정리.
6. 동일한 역확산 noise로 각 sampling step 및 최종 이미지 값.

수치 기본 기준은 abs 또는 rel 1e-3이며 NaN/Inf는 실패다.
불일치 시 최초 분기 블록·parameter·step을 기록하고 해당 성능 결과 공개를 중단한다.
허용치 확대나 ignore로 통과시키지 않는다. Sigmoid의 알려진 수식 차이는 별도 이슈로
보존한다. 기준 DDPM에는 SiLU가 사용되므로 MLP 이슈를 DDPM 전환의 선행 차단점으로
삼지 않되, 기존 전체 P1의 MLP 동등성 항목이 해결됐다고 주장하지 않는다.

release benchmark는 기준 DDPM을 중심으로 연산, 모델 forward/backward,
Adam을 포함한 Trainer E2E, sampling을 구분한다. 초기화와 경로 변환 비용을 별도 기록하고
warmup·반복 횟수·median·p95·throughput·양쪽 live storage/graph 수를 남긴다.
계산 graph의 표현이 다르므로 노드 수 자체의 일치를 요구하지 않는다.
하드웨어·컴파일러·feature·기준 commit·변경 hash·fixture hash·명령을 함께 저장한다.

## 8. 구현 단계와 종료 조건

1. **기준 고정:** 현재 테스트 재실행, 원본 hash 검증, 위 DDPM의 연산/parameter/학습
   의존 지도와 재생 fixture 작성. 종료: 원본 기준 실행과 결과를 재현할 수 있음.
2. **실행 경계 검증:** 현재 Context 모델을 기준으로 한 블록의 Legacy 전달 adapter를 검증.
   Trainer 및 상위 계층의 직접 내부 의존을 감사하고, 공개 API로 옮길 책임을 식별한다.
   기존 계약을 그대로 사용하는 안과 기계적 API 치환안을 먼저 검증한다. 두 안으로
   충분한 부분은 재설계 대상에서 제외하고, 해결되지 않는 의미 차이만 adapter 설계에 반영한다.
   종료: 기존 Context 모델 호출을 유지하는 공개 API 영향·legacy graph 소유 규칙 확정.
3. **경로 연결:** 공통 실행 진입점, 생성 옵션, parameter snapshot, 실제 경로 추적,
   미등록/feature 제외 오류를 구현. 종료: 옵션만 바꿔 같은 블록과 backward 실행.
   공개 학습 API 아래 두 adapter를 연결하고 동일 Trainer의 최소 학습 step도 검증한다.
4. **기준 DDPM 연결:** 확정 구조로 전체 모델을 연결하고 블록→U-Net→loss→Adam 순서로
   검증. 종료: baseline 포함 3자 비교와 기준 3 epoch 조건 통과.
5. **학습·수명 통합:** loader, hook, clipping, capture, interrupt, checkpoint 계약 연결.
   종료: 성공/실패 cleanup·이벤트 순서·foreign/nested 요청 검증, sampling 동등성 통과.
6. **모듈 교체 검증:** builtin 구현을 실제 제외한 빌드와 외부의 실질적 대체 구현으로
   동일 모델 계약 실행. 무거운 DDPM 전체 대체가 어렵다면 부족한 capability를 명시하고
   소형 모델 통과만으로 DDPM 교체 검증을 완료 처리하지 않음.
7. **비교와 문서:** parity 통과 항목만 성능 보고, 전환 명령·호환 범위·미지원 동작 공개,
   기존 파일럿 회귀 검사. 종료: 사용자 재작성 없이 경로 전환을 재현 가능.

단계별로 관련 회귀 검사와 no-default, enableBackward, enableVisualization,
all-features 빌드를 검증한다. legacy가 제외된 경우, Context 기본 구현이 제외된 경우,
둘 다 제공되는 경우를 각각 포함한다. 기존 ignore 외 새 ignore는 허용하지 않는다.

## 9. 새 완료 판정

- 기준 DDPM 설정과 사용자 학습 진입점을 유지한 채 옵션 하나로 경로를 전환한다.
- 동일한 이미 구현된 Trainer가 공개 추상화 API만 사용하여 양쪽 경로를 실행한다.
  경로 전환·규격 준수 구현 추가에 Trainer 수정이나 별도 Trainer 구현이 필요하지 않다.
- 기존 P1 구현을 감사·재사용하고, 기계적 연산 API 치환으로 충분한 상위 코드는
  재작성하지 않는다. 추가 구조 변경에는 최소 보완으로 해결할 수 없는 근거가 있다.
- 모델·레이어·optimizer·loader·observer 등 상위 계층에도 내부 구현 의존이 없으며,
  외부 공개 API 테스트와 의존성 검증으로 이를 확인한다.
- 실제 실행되는 storage/forward/backward 경로가 요청과 일치함을 검증한다.
- 현재 Context Diffusion을 공통 모델로 사용하고 기본 P1/명시적 Legacy 선택을 제공한다.
- 원본과 adapter, adapter와 Context의 수치·Adam 3 epoch·sampling 비교가 통과한다.
- 외부 모델 코드에서 kernel/graph/registry 접근 없이 사용하고, 하위 구현 교체·제거가
  동일 계약 내 정상 실행 또는 지정 오류로 귀결된다.
- cleanup·공유 parameter·observer·checkpoint 검증과 feature matrix가 통과한다.
- Context production 경로에 legacy 전역 storage, tensor storage unsafe,
  실행 중 unwrap/expect/todo/unimplemented가 없다. 보존 원본의 내부 특성을
  Context 경로에 전파하지 않으며, legacy adapter의 한계는 별도 명시한다.
- 레거시와 재실행 가능한 비교 도구를 계속 보존한다. 현재 성능 저하를 숨기지 않는다.

기준 DDPM 전환 완료와 전체 P1 완료는 구분한다. 기존 MLP 수식 불일치 등 남은
수용 항목은 P1_STATUS에서 계속 추적하며, 이번 계획 변경으로 자동 완료하지 않는다.

## 10. P1 완료 후 — ExecutionContext 정적 그래프 전환

추가 사용자 결정: **모델 학습 시작 전에 실행 준비 단계를 둔다.** 그래프를 구성한 뒤
forward와 backward를 함께 분석하여 값의 수명·참조 관계·마지막 사용 시점,
보존해야 할 중간값, 버퍼 배치·재사용, 필요한 복사를 계산한다. 여기서 사전 계산은
텐서 값의 계산이 아니라 실행 흐름과 메모리 사용 계획의 계산이다.

준비된 계획은 반복 배치에서 재사용한다. 입력은 참조로 연결하고 연산 결과는
계획에 배정된 버퍼에 직접 기록한다. 입력이나 중간값을 보존해야 해도 원래 버퍼를
마지막 사용까지 유지할 수 있다면 복사하지 않는다. 덮어쓰기와 이전 값 보존이
충돌하거나 커널의 데이터 배치 요구 등으로 필요한 경우에만 복사를 계획한다.
입력 메모리 수명 보장이 어려운 경우에는 실행 소유 버퍼로 최초 전달 복사를 허용한다.

실행 중에는 계획의 shape·alias 조건과 참조 유효성을 확인하고, 데이터에 따라
달라지는 분기 등 실제 실행 시에만 결정 가능한 부분을 처리한다. 계획 조건이
바뀌면 호환되는 기존 계획을 선택하거나 실행 전에 재분석한다. 유효하지 않은
계획으로 버퍼를 재사용하지 않는다. 따라서 기본 방향은 사전 분석·계획이며,
실행 중 흐름 조정은 필요한 부분으로 제한한다. 이 준비 단계는 공개 실행 API
아래에서 제공하고 기존 Trainer 알고리즘에 메모리 관리 책임을 추가하지 않는다.

사용자 결정: 현재 P1의 완료 조건을 먼저 충족한 뒤, ExecutionContext의 정적
그래프 실행을 별도 후속 단계로 추진한다. 정적 그래프 전환을 P1의 선행 조건으로
추가하거나 현재 경로 연결 작업과 함께 runtime 전체를 재작성하지 않는다.

현재 Context는 forward 중 연산을 즉시 실행하고 gradient record를 생성하는
동적 그래프다. backward와 학습 scope의 cleanup은 그래프 수명을 관리하며,
다음 배치에서 사용할 정적 실행 계획이나 버퍼 재사용을 구현한 것은 아니다.
명시적인 context 소유권과 공개 실행 API를 정적 실행의 기반으로 재사용한다.

후속 단계의 범위:

- 연산 구조와 forward/backward 실행 순서를 구성해 반복 배치에서 재사용한다.
- 그래프 구조와 배치별 값을 분리하고 입력·중간값·gradient 버퍼의 수명과
  재사용을 설계한다. 이전 배치 참조 누적 방지는 별도로 검증한다.
- shape 또는 제어 흐름이 바뀔 때 재구성·캐시·미지원 처리 기준을 정한다.
- 기존 공개 모델/Trainer API와 provider 교체 계약을 최대한 재사용하며,
  정적 실행의 수치 동등성, cleanup, 반복 실행 메모리와 성능을 검증한다.

복사 비용은 시각화와 구분한다. `contracts::snapshot` 및 runtime의 입력 snapshot은
owned TensorBuffer를 만드는 일반 데이터 복사로, `enableVisualization`이 꺼져도
forward/backward에서 호출된다. 시각화용 graph/backward snapshot과 다른 기능이다.
이 복사가 전체 실행 시간에서 차지하는 비중은 아직 측정되지 않았다. 후속 단계에서
시각화 비활성 조건으로 측정해 필요성을 판단하고, 시각화 활성 시에만 발생하는
추가 비용은 이번 요청에 따른 필수 최적화 대상으로 삼지 않는다.
