# Trainer API 전환 안내

학습 방식은 Trainer에서 선택하고, 모델은 예측을, loss는 수치 손실 계산을 담당한다. 기존 Objective 인자와 `TrainerBuilder`를 사용하는 호출부는 아래 방식으로 전환한다.

## 기본 호출

```rust
use trench_deep::loss::MseLoss;
use trench_deep::trainer::{EpochSchedule, Trainer};

let loss = MseLoss::new();
let trainer = Trainer::supervised(&ctx).minimal().with_seed(42);
let result = trainer.fit(&mut model, &loss, &mut optimizer, input, EpochSchedule::new(10)?)?;
```

Optimizer에 모델 파라미터를 등록하는 책임은 호출부에 유지한다. `fit_checkpointed`에도 모델 다음에 `&loss`를 전달한다.

| 학습 방식 | 생성 API | 현재 지원 모델 |
| --- | --- | --- |
| 지도 | `Trainer::supervised(&ctx)` | `ForwardModel` 구현 모델 |
| 자기회귀 | `Trainer::autoregressive(&ctx)` | `BigramLm` |
| diffusion | `Trainer::diffusion(&ctx)` | `Diffusion` |
| 준지도 | `Trainer::semi_supervised(&ctx)` | `PiClassifier` |

모델 타입에 학습 방식을 고정하지 않는다. 예를 들어 `PiClassifier`는 일반 지도학습에도 사용할 수 있다. 잘못된 모델·배치 조합은 trait 제약으로 거부한다. 내부 학습 방식 판별과 observer·checkpoint 메타데이터는 `ParadigmTag` enum을 사용한다. diffusion의 checkpoint 태그는 기존 `Unsupervised`를 유지한다.

## 프리셋과 개별 설정

```rust
let trainer = Trainer::supervised(&ctx)
    .verbose()
    .metrics(Metrics::none().grad_norm())
    .show_progress(false)
    .with_max_grad_norm(1.0)?
    .prepared();
```

| 메서드 | 배치 진행 갱신 | 완료 후 배치 요약 | epoch 로그 | 유한 gradient 검사 | 내장 메트릭 | 진행 표시 |
| --- | --- | --- | --- | --- | --- | --- |
| `.silent()` | 끔 | 끔 | 끔 | 끔 | 없음 | 끔 |
| `.minimal()` | 매 배치 | 끔 | 10 epoch | 매 배치 | 없음 | 켬 |
| `.default()` | 매 배치 | 끔 | 10 epoch | 매 배치 | 학습 방식 대표 메트릭 | 켬 |
| `.verbose()` | 매 배치 | 100배치 | 매 epoch | 매 배치 | 모두 | 켬 |

모든 생성자는 default 설정으로 시작한다. 네 프리셋과 개별 설정은 `.prepared()` 전후 모두 사용할 수 있다. 마지막 호출이 해당 설정을 덮어쓴다. 예를 들어 `.metrics(...).minimal()`은 추가 메트릭을 끄고, `.minimal().metrics(...)`는 지정한 메트릭을 켠다.

프리셋은 로그·내장 메트릭·gradient 검사·진행 표시만 바꾼다. context, seed와 현재 RNG 진행 상태, hook, observer, checkpoint 경로, clipping, noise scale, ramp는 보존한다. `silent`에서도 사용자 hook과 observer는 실행된다.

기존 `Trainer::minimal(&ctx)`는 `Trainer::supervised(&ctx).minimal()`로, `Trainer::builder(&ctx)...build()`는 `Trainer::supervised(&ctx)...`로 바꾼다. `Trainer<Prepared>`는 지도학습의 기본 타입 표기이며 다른 방식은 두 번째 타입 인자에 전략 타입을 가진다.

## Loss와 모델

`MseLoss`, `MaeLoss`, `BinaryCrossEntropyLoss`, `CrossEntropyLoss`, `SoftmaxCrossEntropyLoss`는 `new()`로 만든다. Huber는 `HuberLoss::new(delta)`로 만든다. 기존 ctx 연산에 위임하며 입력 검증, 안정성, target gradient 차단 의미를 유지한다.

Reduction의 최종 API는 후속 논의 대상이다. 현재 `new()`는 기존 예제와 같은 `Mean`을 사용한다. 명시적 비교·검증에는 `.with_reduction(Reduction::Sum)` 또는 `set_reduction(...)`을 사용할 수 있다. 최종 학습 loss는 scalar여야 하며 `None` 결과를 자동 mean으로 바꾸지 않는다.

```rust
let mut loss = HuberLoss::new(1.0);
trainer.fit(&mut model, &loss, &mut optimizer, input, schedule)?;
loss.set_delta(0.5);
// 다음 fit은 바뀐 설정으로 새 그래프를 준비한다.
trainer.fit(&mut model, &loss, &mut optimizer, input, schedule)?;
```

`fit`은 `&loss`를 받는다. 상태 변경은 fit 사이에 명시적 메서드로 수행한다. 한 fit 도중 loss 종류·reduction·delta를 바꾸거나 loss 내부에서 난수를 생성하지 않는다.

사용자 정의 loss는 다음 계약을 구현한다.

```rust
pub trait Loss {
    fn compute(&self, ctx: &ExecutionContext, prediction: &Variable, target: &Tensor)
        -> MlResult<Variable>;
}
```

일반 단일 입력 모델은 `TrainableModel`과 `ForwardModel`을 구현한다. `ForwardModel::forward(&self, input: &Variable)`에는 미분 가능한 예측 연산만 둔다. 분류 모델의 forward는 logits를 반환하며 기존 predict의 확률 반환 의미는 유지한다.

준지도 설정은 다음처럼 지정한다. `PiClassifier::new(&ctx, inputs, outputs)`에는 noise scale을 넣지 않는다.

```rust
let trainer = Trainer::semi_supervised(&ctx)
    .minimal()
    .with_noise_scale(0.1)?
    .with_ramp(ConsistencyRamp::Constant(0.4));
```

기본 noise scale은 0.1, ramp는 30 epoch 동안 0에서 1로 증가하는 기존 sigmoid 설정이다. 준지도 consistency는 기존 차이·제곱·합산 수식을 유지해 두 예측 분기의 gradient를 보존한다. 외부 loss는 지도 항에 적용한다. Diffusion의 feed helper·scheduler·RNG와 자기회귀 시퀀스 이동·유효 token 집계도 유지한다.

## 사용자 정의 학습과 prepared

`TrainingStrategy<M>`는 배치 타입·enum 태그를 선언하고, `forward_batch`에서 loss 참조를 받아 입력 준비·모델 호출·loss·메타데이터를 구성한다. `Trainer::from_strategy(&ctx, strategy)`로 주입한다. Prepared는 추가로 `PreparedTrainingStrategy<M>`가 필요하며 자동 eager fallback은 없다.

- `execution_batch`는 캡처 밖에서 난수·입력·메타데이터를 준비한다.
- `forward_inputs`는 loss 참조를 받아 모델 forward와 loss 수치 연산을 캡처한다.
- epoch별 lambda는 입력 텐서로 전달한다. 수식 구조가 달라지면 variant를 바꾼다.
- shape·variant 캐시는 한 fit 동안 유지되고 다음 fit에서 다시 생성된다.
- Trainer는 loss만 backward root로 사용하면서 prediction을 메트릭에 제공한다.
- eager loader의 그래프 수명과 prepared의 입력 준비/실행 scope 구분을 유지한다.

직접 실행은 같은 내부 adapter를 사용하는 진입점을 이용한다.

```rust
let strategy = Supervised;
let batch = strategy.execution_batch(&mut model, &batch, &TrainingStepContext::default())?;
let mut executor = ctx.prepare_training_for_loss(&mut model, &strategy, &loss, &batch.inputs)?;
```

Prediction도 backward root로 사용할 경우 `prepare_training`을 사용한다. 직접 소유한 executor의 loss 설정은 캡처 시점에 고정되므로 loss 설정 변경 후 재준비해야 한다. `TrainingModel`·`PreparedModel`·`prepare_model*`은 하위 실행 계약으로 남으며 일반 모델이 구현할 필요는 없다.

`TrainingOutput`의 loss·prediction·target·weight·tokens·lambda 형식은 유지한다. `TrainingStepContext`의 epoch/batch는 0부터, observer 번호는 1부터 시작한다. Lambda는 `ParadigmTag::SemiSupervised`일 때만 Trainer ramp에서 공급한다.

## Checkpoint와 RL

`fit_checkpointed`는 모델의 `CheckpointableModel` 구현이 필요하다. checkpoint 경로를 지정한 일반 `fit`은 오류를 반환한다. interrupt 시 모델과 메타데이터를 저장하며 optimizer·RNG·loader를 포함한 완전 재개는 제공하지 않는다.

RL 환경·에피소드 API와 legacy 구현은 유지한다. RL에 공통 로그 설정을 전달할 때는 `RLTrainer::from_trainer(&ctx, Trainer::supervised(&ctx).minimal())`을 사용할 수 있다.

## 검증 인계

이번 변경에서는 테스트·벤치마크를 실행하지 않는다. 아래는 사용자가 실행할 검증 명령이다.

```powershell
cargo test --features enableBackward --lib trainer::preset_tests
cargo test --features enableBackward --test loss_objectives --test unified_trainer --test prepared_model --test diffusion_routes
cargo test --all-features
```

프리셋 값과 설정·RNG 보존, six-loss 연산/gradient/오류 동등성, loss 교체·setter·scalar 제약, 양쪽 consistency gradient, epoch별 lambda, token 집계, eager/prepared 동등성, 캐시·오류 정리, hook·observer·clipping·checkpoint를 검증하는 코드를 전환·추가했다. 그래프 노드·slot·backward root·arena 비교도 포함한다.

성능 비교는 같은 모델·loss·입력·seed를 사용하고 준비 시간과 반복 실행 시간을 분리한다. 프리셋별 메트릭 비용이 섞이지 않도록 비교 양쪽의 설정을 맞춘다. 실행 시간 개선이나 무손실 성능은 측정 전에는 확정하지 않는다.
