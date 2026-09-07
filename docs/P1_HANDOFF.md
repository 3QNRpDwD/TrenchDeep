# 새 세션 인계 — 2026-09-07

## 후속 작업 (동일 날짜)

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

- 기준은 레거시 `diffusion_train_with_trainer`의 DDPM이다. 같은 사용자 진입점에서
  옵션만 바꿔 legacy/ExecutionContext 경로를 빠르게 선택하는 것이 목표다.
- Trainer·모델·레이어·optimizer·loader·observer 등 상위 계층은 공개 추상화 API만
  사용한다. 경로 교체 때문에 기존 Trainer를 다시 구현하거나 수정하면 안 된다.
- 기존 P1 구현을 먼저 감사·재사용한다. 기계적 연산 API 치환으로 충분하면 재작성 금지.
- 모델 소스 통합/분리 방식은 **아직 미확정**이다. 단계적 연결은 권고일 뿐 확정안이 아니다.
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

실행 순서는 블록 adapter 경계 검증 → 모델 공유 범위 결정 → 전체 경로 연결과
구조적 parameter 매핑 → 공통 Trainer·sampling·수명 검증 → parity 통과 후
benchmark다. 외부 provider/feature 제외 및 observer 실패 조합 검증도 포함한다.
MLP 항목은 수식 한 줄의 문제가 아니라 기준 변경과 비교 의미에 대한 별도 결정이 필요하다.
이번 소규모 정리에서는 전체 runtime/모델 설계 변경을 시작하지 않았다.

## 새 세션 시작 순서

1. 이 파일 → `P1_REVISED_PLAN.md` → `P1_DEPENDENCY_AUDIT.md` → `P1_STATUS.md` 순서로 읽는다.
   다른 문서의 원본 불변/adapter 선택/과거 테스트 수는 이 파일의 최신 결정보다 우선하지 않는다.
2. git status/diff를 확인하고 아래 관련 검사를 실행한다. 레거시 파일을 일괄 format하지 않는다.
3. 기존 facade/provider와 공개 scope를 재사용하여 full legacy 실행 연결의 작은 블록 경계를
   검증한다. 모델 통합/분리 결정 전 전체 모델/Trainer 재작성은 하지 않는다.
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
