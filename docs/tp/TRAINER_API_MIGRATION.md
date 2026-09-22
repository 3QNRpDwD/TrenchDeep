# Trainer API 통합 및 전환 안내

작성일: 2026-09-22

## 적용 범위

배치 학습의 공개 진입점을 `Trainer<Mode = Eager>`로 통합했다. 지도·비지도·준지도·자기회귀 학습은 모델의 `TrainingModel` 구현으로 구분하고, eager/prepared 실행 방식은 타입으로 선택한다.

기존 API와의 호환 어댑터는 제공하지 않는 breaking change다. RL의 환경·에피소드 학습 구조와 완전한 checkpoint resume 구현은 이번 변경 범위에 포함하지 않는다.

## 호출부 전환

| 이전 API | 새 API |
| --- | --- |
| `Trainer::builder()` | `Trainer::builder(&ctx)` |
| `TrainerBuilder::new()` | `TrainerBuilder::new(&ctx)` |
| `Trainer::silent().supervised(&ctx)` | `Trainer::silent(&ctx)` |
| `.unsupervised(&ctx)`, `.semi_supervised(&ctx)`, `.autoregressive(&ctx)` | 제거. 모델의 `TrainingModel` 구현으로 구분 |
| `SupervisedTrainer::new(&ctx)` 등 개별 Trainer 생성 | `Trainer::new(&ctx)` |
| `Trainer::silent().prepared(&ctx)` | `Trainer::silent(&ctx).prepared()` |
| `PreparedTrainer` 타입 표기 | `Trainer<Prepared>` |
| `fit_loader(...)` | `fit(...)` |
| 배치 Trainer의 미구현 `resume(...)` | 제거. 재개 기능은 별도 작업 |

`new(&ctx)`는 silent 프리셋을 사용한다. `silent`, `minimal`, `default`, `verbose`는 모두 context를 인자로 받으며 기존 프리셋 설정값을 유지한다. `TrainerBuilder`의 인자 없는 `Default` 구현은 제거했다.

다음은 이미 준비된 model·optimizer·input을 사용하는 호출 예시다.

```rust
use trench_deep::trainer::{EpochSchedule, Trainer};

let trainer = Trainer::builder(&ctx)
    .seed(42)
    .show_progress(false)
    .build(); // Trainer<Eager>

let result = trainer.fit(
    &mut model,
    &mut optimizer,
    input,
    EpochSchedule::new(10)?,
)?;
```

prepared 실행은 생성한 eager Trainer를 소비하여 선택한다. 설정·hook·observer·gradient clipping·준지도 ramp는 전환 후에도 유지된다.

```rust
let trainer = Trainer::minimal(&ctx)
    .with_max_grad_norm(1.0)?
    .prepared(); // Trainer<Prepared>

let result = trainer.fit(
    &mut model,
    &mut optimizer,
    input,
    EpochSchedule::new(10)?,
)?;
```

두 타입 모두 `with_seed`, `with_hook`, `with_observer`, `check_finite_gradients`, `with_max_grad_norm`, `with_ramp`를 제공한다. 이 메서드들은 `.prepared()` 전후 모두 호출할 수 있다.

`fit`의 인자 순서와 `TrainResult`는 동일하다. eager는 `M: TrainingModel`, prepared는 `M: PreparedModel`을 요구하며 입력에는 `IntoBatchLoader<Batch = M::Batch>` 제약을 적용한다. prepared trait을 구현하지 않은 모델은 컴파일 단계에서 거부한다. provider나 연산의 prepared 지원 부족은 실행 중 오류로 반환하며 eager로 자동 전환하지 않는다.

## 모델 구현 전환

`TrainableModel`의 context·parameter 계약은 유지한다. 기존 `SupervisedModel`, `UnsupervisedModel`, `SemiSupervisedModel`, `AutoregressiveModel` 대신 다음 계약을 구현한다.

```rust
pub trait TrainingModel: TrainableModel {
    type Batch: BatchInputs;
    const PARADIGM: &'static str;

    fn forward_batch(
        &mut self,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<TrainingOutput>;
}
```

기존 모델의 `forward_loss` 수식은 모델의 일반 메서드로 유지할 수 있다. `forward_batch`는 이를 호출하고 학습 결과의 메타데이터를 구성한다. 저장소의 기존 pilot 모델은 이 방식으로 전환했다.

| 패러다임 | `Batch` | `PARADIGM` | 집계·메트릭 규칙 |
| --- | --- | --- | --- |
| 지도 | `SupervisedBatch` | `"supervised"` | sample 수를 weight로 사용하고 target을 전달 |
| 비지도 | `UnsupervisedBatch` | `"unsupervised"` | sample 수를 weight로 사용 |
| 준지도 | `SemiSupervisedBatch` | `"semi_supervised"` | labeled sample 수, labeled target, 현재 lambda 전달 |
| 자기회귀 | `AutoregressiveBatch` | `"autoregressive"` | 유효 token 수를 weight와 tokens에 전달 |

현재 문자열 값은 준지도 ramp와 패러다임 메트릭 선택에 사용되므로 위 표기와 일치시켜야 한다.

`TrainingOutput`은 다음 정보를 담는다.

| 필드 | 의미 |
| --- | --- |
| `loss: Variable` | 역전파할 scalar loss |
| `prediction: Option<Variable>` | accuracy·hook 등에 제공할 예측 |
| `target: Option<Tensor>` | 예측과 비교할 정답 |
| `weight: usize` | epoch loss의 가중 평균에 사용할 양수 개수 |
| `tokens: Option<usize>` | 자기회귀 학습의 유효 token 수 |
| `lambda: Option<f32>` | 준지도 학습 메트릭·hook에 전달할 현재 가중치 |

`TrainingStepContext`의 `epoch`, `batch`는 **0부터 시작**한다. 기존 observer의 epoch·batch 번호는 **1부터 시작**하는 규칙을 유지한다. `lambda`는 `PARADIGM == "semi_supervised"`일 때만 `Some(ramp.value(epoch))`이며 나머지는 `None`이다. 모델은 이를 실제 loss 계산에 사용하고 결과 메타데이터에도 전달해야 한다.

## PreparedModel 전환과 실행 수명

`PreparedModel`은 이제 `TrainingModel`을 상속한다. `Batch`와 `PARADIGM` 선언은 `TrainingModel` 구현으로 옮기고, `execution_batch`에 step 인자를 추가한다.

```rust
pub trait PreparedModel: TrainingModel {
    fn execution_batch(
        &mut self,
        batch: &Self::Batch,
        step: &TrainingStepContext,
    ) -> MlResult<PreparedBatch>;

    fn forward_inputs(&self, inputs: &ExecutionInputs) -> MlResult<ModelOutput>;
}
```

- `execution_batch`는 매 배치 호출된다. 난수 생성·입력 준비·스케줄 값 처리는 이 단계에서 수행한다.
- `forward_inputs`에는 캡처할 수치 연산을 둔다. 입력 signature와 topology variant가 같으면 기존 executor를 재사용한다.
- epoch마다 바뀌는 lambda는 `ExecutionInputs`의 텐서로 전달한다. Rust의 scalar 값을 캡처된 수식에 고정하면 이후 epoch의 lambda 변경이 반영되지 않는다.
- 수치 연산 구조가 달라지면 variant를 변경한다. 동일 구조에서 입력 값만 바뀌는 경우에는 variant를 변경할 필요가 없다.
- 캐시는 한 번의 `fit` 안에서 epoch를 넘어 재사용하며, `fit`이 끝나면 폐기한다. 다른 shape나 variant는 별도 executor를 사용한다.
- eager는 loader가 만든 계산 그래프를 backward까지 유지한다. prepared는 loader·입력 준비와 executor 실행 scope를 분리한다. loader 내부의 미분 가능한 연산이 자동으로 prepared 그래프에 편입되는 것은 아니다.

`prepare_model`, `prepare_model_for_loss`, `PreparedModelExecutor::run`을 사용하는 직접 실행 경로는 유지한다. 직접 `execution_batch`를 호출하는 코드는 step을 명시해야 한다.

```rust
let step = TrainingStepContext {
    epoch: 0,
    batch: 0,
    lambda: None,
};
let batch = model.execution_batch(&batch, &step)?;
let mut executor = ctx.prepare_model_for_loss(&model, &batch.inputs)?;
```

준지도 모델에서는 위 예시의 `lambda`를 현재 ramp 값으로 지정한다. `TrainingStepContext::default()`의 lambda는 `None`이다.

기존 `Diffusion`과 prepared MLP 예제·벤치마크의 계약을 전환했다. 이번 통합이 모든 기존 모델에 prepared 지원을 자동으로 추가하는 것은 아니다. eager 전용 모델은 계속 `TrainingModel`만 구현할 수 있다.

## 체크포인트와 RL

`fit_checkpointed`는 해당 실행 모드의 모델 계약에 더해 `CheckpointableModel`을 요구한다. checkpoint 디렉터리를 설정한 채 일반 `fit`을 호출하면 명시적 오류를 반환한다. 저장은 기존과 같이 interrupt 처리 시 수행하며, 모델 가중치와 메타데이터 저장을 optimizer·RNG·loader까지 포함한 완전 재개로 해석하면 안 된다.

`RLTrainer`의 `fit`·환경·에피소드 API는 유지한다. custom Trainer를 전달하는 코드는 context를 생성 시에도 지정한다.

```rust
let trainer = RLTrainer::from_trainer(
    &ctx,
    Trainer::builder(&ctx).seed(42).build(),
);
```

legacy benchmark 전용 Trainer 구현은 유지했다. legacy와 현재 API를 함께 비교하는 벤치마크에서는 현재 API 호출부만 새 이름으로 전환했다.

## 검증 인계

사용자 요청 이후 테스트를 추가 실행하지 않는다. 아래 명령과 확인 항목은 직접 검사를 위한 안내다.

```powershell
cargo test --features enableBackward --test unified_trainer --test prepared_model
cargo test --features enableBackward
cargo check --all-targets --all-features
```

`tests/unified_trainer.rs`에는 다음 검증을 추가했다.

- eager/prepared의 가중치·loss·gradient norm·observer 이벤트 일치
- 설정·hook·clipping 보존과 epoch별 lambda가 실제 gradient에 반영되는지 확인
- variant별 캐시 재사용, 별도 fit 호출 간 캐시 분리
- 입력 준비 실패 후 graph·gradient 정리와 다음 학습 복구
- context 불일치와 optimizer parameter 불일치 거부
- checkpointed fit 요구와 정리 이후 checkpoint 저장

기존 `tests/prepared_model.rs`는 shape별 캐시와 eager loader 그래프의 gradient 연결을 검증한다. `TrainingModel` 문서에는 eager의 정상 컴파일 예시와 prepared trait 부족 시 compile-fail 예시를 추가했다.

요청 전 실행한 집중 테스트는 7개가 통과했다. 중단 요청 전에 시작한 전체 테스트와 모든 타깃 컴파일 검사는 이후 로그 확인 시 이미 종료되어 있었다. 기록된 전체 테스트 결과에는 실패가 없고, 컴파일 로그에는 완료가 기록되어 있다. 이는 모든 feature 조합의 런타임 테스트를 수행했다는 의미는 아니다. 기존 경고는 남아 있다.

기존 실행 로그는 `target/trainer-focused.log`, `target/trainer-test.log`, `target/trainer-check.log`에 있다. `target` 아래 로그는 로컬 산출물이므로 저장소 배포 문서의 일부가 아니다.
