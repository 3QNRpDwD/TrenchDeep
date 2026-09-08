# 새 세션 인계 — 2026-09-07

## 2026-09-08 일괄 연산 연결

후속 사용자 승인 수정: Tanh/Softmax backward가 입력에서 출력을 재계산하도록
src와 legacy/src 양쪽을 수정했다. Softmax는 SoftmaxCrossEntropyLoss와 같은
max 차감 exp 정규화를 기존 forward로 재사용하며 일반 VJP를 유지한다.
축 gradient shape도 입력 속성 shape에 맞췄다. 두 연산의 tracked route가 열렸다.
사용자 Sigmoid 수정에는 누락된 참조만 보완했다. CORRECTIONS.json 갱신 완료.
아래 tracked Softmax 미지원 기록은 이 수정 이전 상태다.

최신 지원표는 `P1_LEGACY_OPERATIONS.md`를 따른다. Pow/근사 삼각함수,
Concat, batched Matmul 일부, Conv2d/GroupNorm/pooling/upsample,
6종 mean loss와 다중 출력 추론을 연결했다. 저장 텐서는 원본 API로 등록하며
loss target은 별도 native leaf로 분리한다. 아래 shape/matrix 단계는 이전 기록이다.
전체 미지원 해소는 아직 아니다. 원본 backward/출력 계약이 다른 연산은 명시적으로
제한하며 원본 수식은 변경하지 않았다. 특히 tracked Softmax가 남아 있어
Context Diffusion의 Legacy 학습 전환 완료로 해석하지 않는다.

## Route 생성 경계 구현 시작

최신 shape/matrix 확장: Reshape(원소 수 동일), Transpose(rank>=2, identity 또는
한 쌍 축 교환), Matmul(빈 차원 제외 1D/2D 조합)을 원본 호출로 지원한다.
batched Matmul과 여러 축 교환이 필요한 순열은 아직 미지원이다. 레거시 Reshape는
목표 shape를 가진 전체 크기 dummy tensor를 요구해 추가 할당하며, 무복사 구현은 아니다.
shape/axis 속성은 adapter가 원본 입력 형식으로 전달한다. 수식 변경 없음.

최신 연산 지원 확장: Sub(동일 shape), Neg / Square, Exp / Sin, Cos /
ReLU, SiLU를 원본 호출로 추가했다. Abs, Log, Sqrt는 추론 전용이며 tracked
호출은 graph 생성 전에 UnsupportedCapability를 반환한다. 따라서 아래 Add/Mul만
지원한다는 기록은 이전 단계다. Sum, Tanh, Div는 원본 계약 검증이 필요해 보류했다.
원본 수식 수정 없음. 연산별 세부 제한은 P1_STATUS의 최신 항목을 따른다.

최신 연결 상태: `legacyBenchmark` 빌드에서 `.route(ExecutionRoute::Legacy).build()?`
가 실제 세션을 생성한다. 공개 tensor/view, Add/Mul, native backward, gradient 조회,
parameter replace와 공통 with_training_scope가 연결됐다. 동일한 외부 모델·기존
UnsupervisedTrainer·SGD의 3 epoch 결과가 P1과 일치한다. P1 기본 구현을 제외한
Legacy-only 구성도 별도 검사한다. 아래의 “공개 route 미지원”은 이전 상태다.
미지원: Add/Mul 이외 연산, custom op, detach alias, explicit backward seed, capture.
Legacy와 사용자 지정 P1 provider 혼합도 명시적으로 거부한다. DDPM/Adam 전환은
아직 완료되지 않았다. 원본 graph가 thread-local 공유이므로 세션은 thread당 하나며
동일 thread에서 raw legacy API를 동시에 사용하지 않는 계약을 유지한다.

Native 세션 후속 구현: 원본 forward만 호출하는 nested no-grad, shape 검증 후
원본 tensor replace, RAII training_step을 추가했다. 16회 갱신에서 값/gradient와
tensor/graph 수 원복, 기존 graph 중 no-grad의 graph 보존, 오류/panic 후 재사용을
검증했다. 원본 연산 수식은 변경하지 않았다. 현재 내부 training_step은 owned
buffer만 반환하고 step 안에서 만든 native handle을 정리하는 실험 계약이다.
공개 with_training_scope의 반환 handle 수명 계약과는 아직 연결하지 않았다.
공개 Legacy route는 계속 미지원이며 ctx/Trainer E2E 완료가 아니다.

후속 native session 실험: `src/runtime/legacy_session.rs`에서 원본 Variable,
Add/Mul, 원본 graph backward를 직접 호출한다. 세션별 handle 검증, thread-local
독점, 기존 그래프 충돌 거부, graph/gradient 정리와 drop을 구현했다. 계산 수식은
추가하지 않았다. 아직 공개 ctx 저장소/handle API에 연결되지 않은 내부 실험이다.
따라서 Legacy route는 계속 UnsupportedCapability를 반환한다. 다음은 이 세션을
공개 tensor 조회·parameter 갱신·scope와 연결하고 no-grad 및 연산 범위를 확대하는 작업.
raw legacy API를 세션 실행 중 직접 호출하는 혼용은 지원하지 않는다. 그 경로까지
가로채는 전역 소유권 보장은 없으므로 공개 route 개방 전에 경계 정책을 검증해야 한다.

`ExecutionRoute::{P1, Legacy}`와 builder의 `.route(...).build()?`를 추가했다.
기존 `ExecutionContext::new()`와 `.build()`는 그대로 P1이며 기존 provider 구성을
유지한다. route는 provider/seed 설정 후 지정한다. Legacy native adapter는 아직
미구현이므로 feature 미포함이면 DependencyUnavailable, 포함됐으면
UnsupportedCapability를 반환한다. P1으로 자동 대체하지 않는다.
이 단계는 생성 계약만 구현한 것이며 Legacy 실행이나 Diffusion route parity가 아니다.
다음 작업은 원본 세션의 handle/forward/backward/cleanup 전달 구현이다.

## 2026-09-08 parameter 매핑 갱신

기준 DDPM 비교는 이제 `Unet::named_parameters()`와 레거시의 읽기 전용 구조
열거를 이름·shape·공유 관계로 검증한다. 아래의 positional 매핑 미완료 기록은
이 변경 이전 상태다. fixture는 version 2이며 초기값에 이름/shape/shared_with,
gradient와 갱신값에 이름 key를 저장한다. version 1 파일은 다시 생성해야 한다.
공유 관계는 실행별 ID를 직접 비교하지 않고 같은 ID를 참조하는 구조적 이름 집합으로
비교한다. 기준 DDPM에 없는 공유 패턴은 별도 매핑 회귀 검사로 검증한다.
레거시 원본 수정 없이 build.rs가 OUT_DIR 사본에 열거 메서드만 추가한다.
이는 전체 실행 경로 adapter나 공통 Trainer 전환의 완료를 의미하지 않는다.

## 후속 작업 (동일 날짜)

- 최소 영향 작업으로 반복 배치 수명 검사를 추가했다. 공개 scope에서 64회
  성공 backward/forward 실패를 섞어 실행하고 매번 graph·gradient 정리 및
  반환 텐서 drop 후 tensor 수 원복을 확인한다. 내장/독립 provider 양쪽 통과.
  cleanup 검사 all-features 5개, no-default 4개 통과. runtime 변경 없음.
  이는 live handle 검사이며 실제 allocator 메모리나 복사량 검증은 아니다.

- `examples/replay_reference_diffusion.rs` 추가: legacy feature 없이 완전한
  `target/p1/reference-draws/fixture.json`을 읽어 원본 예측/loss/모든 gradient/
  Adam 3 step 가중치를 검증한다. 초기 가중치와 보정 provenance도 파일에 저장한다.
- `tests/legacy_reference_diffusion.rs`가 기존 draw 파일에 더해 위 fixture를
  생성하고 파일 round-trip 재생 및 손상된 예측 거부를 검증한다.
- 실행: `cargo run --features enableBackward --example replay_reference_diffusion -- target/p1/reference-draws/fixture.json`.
  기존 `step-N.json`은 noise 기록만 있어 이 CLI 입력이 아니다.
- 구조적 parameter 매핑은 여전히 미완료다. 전체 경로 전환이나 블록 adapter를
  구현했다고 해석하지 않는다. 상세 실행/cleanup 경계 감사는 DEPENDENCY_AUDIT에 추가했다.
- 재생 입력의 모든 step을 학습 전에 검사한다: timestep, tensor 길이,
  parameter별 길이, 유한값. CLI는 입력 경로 하나만 받는다.

저장소: `C:\Users\2qnrp\RustroverProjects\TrenchDeep` (PowerShell).
확인 당시 HEAD `f4cd4fd`. 아래 작업은 **커밋되지 않은 워크트리 변경**이며,
untracked 파일도 포함한다. reset/clean/전체 덮어쓰기를 하지 말고 보존한다.

## 사용자 요구 — 최신 결정 우선

- 구현 기준은 현재 `src/nn/diffusion.rs`의 Context Diffusion/U-Net이다.
  명시적으로 ctx를 생성해 모델·Trainer에 전달하는 구조를 유지한다.
  기본 생성은 P1 경로이며 Legacy가 필요할 때만 생성 시 route를 명시한다.
  레거시 원본 DDPM과 기존 fixture는 수치 비교 기준으로 보존한다.
- Trainer·모델·레이어·optimizer·loader·observer 등 상위 계층은 공개 추상화 API만
  사용한다. 경로 교체 때문에 기존 Trainer를 다시 구현하거나 수정하면 안 된다.
- 기존 P1 구현을 먼저 감사·재사용한다. 기계적 연산 API 치환으로 충분하면 재작성 금지.
- 같은 Context 모델 계산 코드를 사용하고 Legacy adapter는 원본 연산 요청을
  전달한다. 원본 내부 연산 순서·수식·backward 흐름은 변경하지 않는다.
  handle·오류·수명 경계만 연결하며 미지원/미포함 Legacy를 P1으로 대체하지 않는다.
- CPU는 기존 backend를 재사용한다. 레거시 코드/비교 도구는 삭제하지 않는다.
- **최신 명시적 결정:** Conv2D 오류는 별도 보정 adapter가 아니라 원본 레거시에서
  직접 수정하고, 수정된 원본과 비교한다. 직전의 보정 adapter 선택은 폐기됐다.
  이것은 확인된 Conv2D 수정의 승인이지 다른 원본 변경을 포괄 승인한 것은 아니다.

## 이번 워크트리에 구현된 내용

1. `src/runtime/training.rs`: 공개 `with_training_scope` closure API 추가.
   기존 private guard를 재사용하며 graph/gradient 정리, 중첩 거부, 복합 오류 의미 유지.
   `src/trainer/service.rs`, `reinforcement.rs`는 호출부만 치환했다.
2. RL 비공개 연산 helper를 기존 receiver API로, RL·optimizer의 buffer 필드 접근을
   공개 accessor로 기계적 변경했다. Trainer 알고리즘/모델을 새로 만들지 않았다.
3. `tests/cleanup.rs`: 외부 공개 scope + 독립 provider/no-default 검사 추가.
   `tests/diffusion.rs`: 기준 구성의 P1 DataLoader→Adam→Trainer 3 epoch 검사 추가.
4. `legacy/benchmark_lib.rs`, `reference_models.rs`: 기존 crate 경로를 유지하는
   가시성 shim으로 원본 Diffusion을 노출. `comparison.rs`의 `ReferenceDiffusion`,
   `diffusion_draw`는 원본 forward가 생성한 noise와 timestep을 graph에서 읽는다.
5. `tests/legacy_reference_diffusion.rs`: 실제 원본 forward의 draw를 P1에 재생하고
   출력/loss/모든 gradient/Adam 3 step 가중치 비교. **공통 Trainer 전환 검사가 아니다.**
   parameter 매핑은 아직 index zip + shape 확인이다. 이름·공유 관계 매핑은 미완료.
   실행 시 `target/p1/reference-draws/step-N.json`을 덮어쓴다. 원본 RNG는 고정되지 않아
   매 실행 draw가 달라진다. 완전한 `fixture.json`도 저장하며 위 CLI로 재생한다.
6. `legacy/src/tensor/operators/conv2d.rs`: `matmul_at_b`의 한 줄을
   `a[i*k+l]` → `a[l*m+i]`로 수정. 수정 전 입력 gradient [3,7], 올바른 값 [4,6].
   `tests/legacy_conv_gradient.rs`로 P1·유한차분·수정된 원본의 일치를 검증했다.
7. `legacy/BASELINE.json`은 과거 기준 `65b4d40` 해시를 유지한다.
   새 `legacy/CORRECTIONS.json`에 승인된 한 파일의 전후 해시/사유를 기록했고,
   `scripts/verify_legacy.py`가 baseline+corrections를 함께 검사하도록 수정했다.
   원본 142파일 중 141개 그대로, 1개 한 줄 수정. 삭제 없음.

## 기준 모델과 검증 상태

원본 위치: `legacy/src/tests/common/model/diffusion/mod.rs:650`.
1채널 8×8, dim=8, multipliers=[1,2], groups=4, down/up attention=[false,false]
(middle attention은 존재). linear scheduler T=10, beta=1e-4..0.02.
0.5 값 sample 두 개, batch=2, shuffle=false, Adam(1e-3,0.9,0.999,1e-8),
3 epoch, tolerance=1e-10. 작은 기존 4×4·SGD fixture와 혼동하지 않는다.

최신 결과: all-features lib/integration **123 pass / 0 fail / 0 ignore**.
레거시 standalone **309 pass / 0 fail / 기존 3 ignore**.
Conv2D 수정 전 DDPM gradient 비교 실패는 수정 후 해결됐다. 허용치 확대/ignore 없음.
no-default cleanup/providers 12개 및 backward/visualization 빌드는
Conv2D 수정 이후 후속 세션에서 재검증해 통과했다.

## 남은 장기 작업

사용자 후속 결정: **P1 완료 후 ExecutionContext를 정적 그래프로 전환한다.**
학습 시작 전에 forward/backward의 수명·참조·보존 값·버퍼 재사용·필요 복사를
분석하는 실행 준비 단계를 둔다. 배치 실행은 준비한 계획을 재사용하고,
shape·alias 조건 변화 시 다른 계획 선택 또는 재분석을 수행한다.
P1 완료의 선행 조건으로 추가하지 않는다. 구체 범위는 `P1_REVISED_PLAN.md` 10절.
현재 Context는 동적 그래프이며, 일반 tensor snapshot 복사는 시각화 feature와
무관하다. 시각화에서만 발생하는 추가 비용의 최적화는 필수 범위가 아니다.

- 동일 Trainer/모델 진입점에서 실제 legacy↔Context 실행 경로 전환.
- 기존 AutogradEngine은 graph 저장/순서 계약이고 backward 실행 자체는 Context에 있다.
  레거시 kernel/VJP를 provider로 연결하는 것만으로 전체 legacy graph 경로를 구현했다고
  주장하면 안 된다. 전체 경로 adapter의 최소 확장 지점을 검증해야 한다.
- 이름·shape·공유 parameter 관계 매핑, 공통 Trainer 기준 E2E 양쪽 비교,
  동일-noise sampling 비교, 수정된 기준의 benchmark.
- 기존 MLP Sigmoid 수식 불일치는 별도 미해결 이슈다. 이번 Conv2D 승인으로 자동 수정하지 않는다.
  과거 benchmark JSON은 **수정 전 원본** 결과이며 최신 성능으로 인용하지 않는다.

실행 순서는 Context 블록의 Legacy 전달 경계 검증 → 전체 route 연결과
구조적 parameter 매핑 → 공통 Trainer·sampling·수명 검증 → parity 통과 후
benchmark다. 외부 provider/feature 제외 및 observer 실패 조합 검증도 포함한다.
MLP 항목은 수식 한 줄의 문제가 아니라 기준 변경과 비교 의미에 대한 별도 결정이 필요하다.
이번 소규모 정리에서는 전체 runtime/모델 설계 변경을 시작하지 않았다.

## 새 세션 시작 순서

1. 이 파일 → `P1_REVISED_PLAN.md` → `P1_DEPENDENCY_AUDIT.md` → `P1_STATUS.md` 순서로 읽는다.
   다른 문서의 원본 불변/adapter 선택/과거 테스트 수는 이 파일의 최신 결정보다 우선하지 않는다.
2. git status/diff를 확인하고 아래 관련 검사를 실행한다. 레거시 파일을 일괄 format하지 않는다.
3. 기존 facade/provider와 공개 scope를 재사용하여 full legacy 실행 연결의 작은 블록 경계를
   검증한다. 현재 Context Diffusion을 재사용하고 전체 모델/Trainer를 재작성하지 않는다.
4. 보정 adapter는 만들지 않는다. 필요 시 다음 설계 선택을 사용자와 논의하되,
   이미 승인된 리팩토링·검증을 다시 허가받을 필요는 없다.

```powershell
git status --short
python scripts/verify_legacy.py
cargo test --features legacyBenchmark --test legacy_conv_gradient --test legacy_reference_diffusion
cargo test --all-features --lib --tests
cargo test --manifest-path legacy/Cargo.toml --lib --features enableBackward
cargo test --no-default-features --test cleanup --test providers
cargo check --no-default-features --features enableBackward
cargo check --no-default-features --features enableVisualization
```

Python이 PATH에 없다면
`C:\Users\2qnrp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` 사용.
전체 검증 로그: `target/p1/continued-*.log`.
소규모 입력 검증 수정 로그: `target/p1/small-fixes-reference.log`.
`target/p1`의 오래된 생성 스크립트는 재실행하지 않는다.
