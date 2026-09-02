
# Trainer Remediation — 진행 보고서

> 2026-09-01 API 통합 결과는 [`TRAINER_API_OPTIMIZATION.md`](TRAINER_API_OPTIMIZATION.md)를 기준으로 한다.

최종 갱신: 2026-04-22 (Phase 3.1 완료 반영)
대상 브랜치: `V4`
테스트 상태: **198 passed / 0 failed / 3 ignored** (`cargo test --lib --features enableBackward`)

이 문서는 `docs/TRAINER_REMEDIATION_PLAN.md` 의 실행 현황을 상세히 기록한다.
플랜 문서는 "무엇을 왜 할 것인가" 를 기술하고, 이 문서는 "무엇이 어떻게
끝났고 다음에 구체적으로 어떤 변경이 들어갈지" 를 기술한다.

**후속 작업 계획**: 이 문서의 §3 "진행 예정" 내용은
`TRAINER_NEXT_PHASES.md` 로 이관되었다. Phase 6/7/8/B1 및 3.1/4.1 의
상세 설계는 그 문서를 참조.

---

## 1. 완료된 작업

### Phase 0 — 스텁 보존 + 컴파일 위생

**목적**: Phase 1 이후 추가로 건드릴 diffusion 스텁들이 경고 없이 컴파일되도록.

**변경**
- `src/tests/common/model/diffusion/decoder.rs` — `Decoder` 에
  `#[allow(dead_code)]` + `TODO(LatentDiffusion)` 주석.
- `src/tests/common/model/diffusion/encoder.rs::Encoder` — 동일.
- `src/tests/common/model/diffusion/scheduler.rs::Scheduler` — 동일.
- `src/tests/common/model/diffusion/mod.rs`
  - `use` 가 제거된 이름(`Decoder`, `Encoder`, `Scheduler` 래퍼) 의 dead `use` 정리.
  - `struct LatentDiffusion;` 에 `#[allow(dead_code)]` + TODO 유지.

**불변 지침 준수**: 사용자가 명시한 "LatentDiffusion/Encoder/Decoder/Scheduler
스텁은 제거 금지" 조건을 그대로 따른다. 내부 `LatentDiffusion` 구현 시 재사용.

### Phase 1 — AutoregressiveTrainer + AutoregressiveModel

**목적**: 자기회귀 모델(언어모델·시퀀스 생성)을 학습할 수 있는 전용
트레이너 도입. 기존 `Trainer` 의 `(x, t)` 계약으로는 표현이 어려움.

**신규**
- `src/trainer/autoregressive.rs`
  - `trait AutoregressiveModel`: `forward_loss(x) → (logits, loss, n_tokens)`.
    `n_tokens` 는 padding 제외한 유효 타깃 토큰 수로, perplexity 계산의
    분모. `predict_raw`, `save_model`, `load_model` 제공 (기본값은
    에러 반환).
  - `struct AutoregressiveTrainer { core: TrainerCore }`
  - `From<Trainer>` 구현으로 기존 프리셋을 그대로 재사용 가능
    (`AutoregressiveTrainer::silent/minimal/default/verbose`).
  - `fit` / `resume` / `fit_inner` (Phase 2 에서 재작성됨).
  - 체크포인트/Ctrl+C 지원.
- `src/tests/common/model/autoregressive/mod.rs`
  - `BigramLM` 토이 모델(vocab-size 전이행렬 `W:[V,V]`) + 파일럿 테스트 2개.
  - 테스트 목적은 AR 트레이너의 제어 흐름 검증이지, 성능 달성이 아님.

**수정**
- `src/trainer/core/metrics.rs` — `Perplexity` 누적기 추가 (`update_loss`,
  `mean_nll`, `reset`). `MetricHook` impl 로 `format() → "PPL: ..."`.
- `src/trainer/core/mod.rs`, `src/trainer/mod.rs` — 재수출 정리.
- `src/tests/common/model/mod.rs` — `pub mod autoregressive`.

**평가**
- `cargo test --lib --features enableBackward`: 177/177 통과.
- AR 파일럿은 학습 진행에 따라 손실이 유의미하게 감소함을 확인.

### Phase B0 — SoftmaxCrossEntropyLoss per-row 버그 수정

**증상**: `[B, V]` (B≥2) 입력에서 전체 플랫 텐서를 단일 분포로 취급해
log-sum-exp 를 계산. B=1 경로에서는 잠복해 있다가 AR 파일럿의 `[L, V]`
시퀀스에서 음수 NLL (`-5.68`) 로 표면화.

**수정**
- `src/loss/function.rs::SoftmaxCrossEntropyLoss::forward`
  - `num_classes = shape.last()`, `n_rows = data.len() / num_classes`.
  - 각 행별로 `(max + ln Σexp(z-max)) - ⟨z, t⟩` 계산.
  - 평균값 `loss_sum / n_rows` 를 스칼라 반환.
- 동 구조체의 `backward`
  - 각 행별 softmax 계산 후 `(p - t)` 를 순서대로 concat.
- 회귀 테스트 4개 추가 (`softmax_ce_tests` 모듈)
  - B=1 호환성, B>1 기본 케이스, one-hot 검증, 수치값 일치.

**측면 결정**
- `BigramLM::forward_loss` 의 per-row 우회 루프는 **유지**. teacher-forcing
  시맨틱을 코드로 드러내는 용도이며, 버그 수정 후에도 의미가 있음.
  배치 최적화는 Phase 2 이후 별도 논의.

**범위 외**
- 다른 손실(`CrossEntropyLoss`, `MSE`, `MAE`, `Huber`, `BCE`) 의 동일 패턴
  버그 여부는 미확인 상태. 별도 B-페이즈로 기록 예정.

### Phase 2 — TrainerCore::run_epoch 공통 루프 추출

**목적**: 4 개 트레이너에 복사된 ~900 라인의 동일 루프를 단일 구현으로 통합.

**신규 모듈**: `src/trainer/core/epoch_loop.rs`

- `struct StepInfo` — 한 배치의 관측치(loss, NaN 플래그, FW/BW 소요, 선택적
  grad_norm/update_ratio, 패러다임별 `extra_msg: Vec<String>`).
- `trait EpochStep` — 패러다임별 배치 훅:
  - `n_batches()`, `forward_backward(batch_idx, cfg) → StepInfo`,
    `optimizer_step()`, `reset_epoch_state()`, `format_epoch_extras(avg_loss)`.
- `struct EpochOutcome` — `avg_loss`, `interrupted`, `epoch_dur`, `summary_extras`.
- `impl TrainerCore { fn run_epoch<S: EpochStep>(&self, step, ...) }` — 배치 바
  생성/완료/실패, NaN 검사 및 에러 전파, 배치 로그 포매팅, optimizer.step,
  Ctrl+C 감지 + `confirm_interrupt()`.

**트레이너 이전**
- `Trainer::fit_inner` → `SupervisedEpochStep<'a, M>` (accuracy 누적 내장)
- `UnsupervisedTrainer::fit_inner` → `UnsupervisedEpochStep<'a, M>`
- `AutoregressiveTrainer::fit_inner` → `AutoregressiveEpochStep<'a, M>`
  (Perplexity 누적 내장; 배치/에폭 로그에 `PPL: ...` 삽입)
- `SemiSupervisedTrainer::fit` → `SemiSupervisedEpochStep<'a, M>`
  (인덱스 wraparound 및 고정 λ 캡처)

**전제/경계**
- RL 은 에피소드 시맨틱이라 범위 외.
- SemiSup 은 원본에 인터럽트/체크포인트 경로가 없어 Phase 4 까지
  `interrupt: None` 으로 유지.

**회귀 영향**
- 배치 로그 포맷에서 **SemiSup 의 `λ` 위치가 변경됨**:
  기존 `FW/BW | λ | GN | UR` → 현재 `FW/BW | GN | UR | λ` (`StepInfo.extra_msg`
  가 맨 뒤에 합류되는 구조). 기능적 변화는 없으나 로그 스냅샷 회귀 감시
  시 인지 필요.
- 에폭 요약은 `AL | LC | <extras> | duration` 의 형태로 통일. AR 은
  `PPL`, SemiSup 은 `λ`, Supervised 는 `AC` 가 extras 로 들어감.

**검증**
- `cargo test --lib --features enableBackward`: 177/177 통과.
- 트레이너별 파일럿 로그의 손실/PPL/accuracy 값이 기존과 동일함을 확인.

### Phase 3 — MetricHook 플러그인 활성화 (코어 완료)

**목적**: `TrainerCore::hooks` 가 저장만 되고 호출되지 않던 문제를 해결.
사용자 정의 메트릭을 배치·에폭 단위로 삽입 가능하게.

**변경**
- `src/trainer/core/metric_hook.rs` — `BatchContext<'a>` 를 완성된
  관측치 구조체로 확장 (`batch_idx`, `pred/target: Option<&dyn TensorBase>`,
  `loss: f32`, `n_tokens: Option<usize>`, `lambda: Option<f32>`, `lr: f32`).
- `src/trainer/core/mod.rs` — `hooks` 를 `RefCell<Vec<Box<dyn MetricHook>>>`
  로 변환. `add_hook(&self)` / `hook_count()` / `clear_hooks()` 추가.
  `&self` 체이닝 스타일 지원 (`trainer.with_hook(…).fit(…)`).
- `src/trainer/core/epoch_loop.rs` — `EpochStep` trait 에 배치별 관측치
  접근자 추가:
  - 필수: `current_lr() -> f32`.
  - 기본 `None` 반환: `last_pred`, `last_target`, `last_n_tokens`,
    `last_lambda`.
  - `run_epoch` 가 `hooks_active` 검사 후 배치마다 `BatchContext` 조립
    → `hook.update(&ctx)` 호출. 에폭 종료 시 `hook.format()` 을
    `summary_extras` 에 합류.
  - 훅 미장착 시 `hooks.borrow().is_empty()` 가드로 zero-overhead.
- 각 `*EpochStep` — 배치별 `last_y: Option<Variable>` 등 stash 필드 추가.
- `src/trainer/core/metrics.rs` — `ClassificationAccuracy` 와 `Perplexity`
  의 `MetricHook` impl 을 새 `BatchContext` 시그니처로 갱신.
- `src/tests/common/model/autoregressive/mod.rs` — `CallCounterHook`
  통합 테스트 추가 (훅이 배치마다 호출되고 에폭마다 reset 됨을 확인).

**테스트**: 181 passed (+4).

**미완** (`TRAINER_NEXT_PHASES.md::Phase 3.1` 으로 분리)
- 프리셋 자동 훅 장착, 인라인 경로 deprecation, bit-identical 골든 테스트.

### Phase 4 — 체크포인트/인터럽트 통합 (부분 완료)

**목적**: 4 패러다임 중 3곳에만 있던 Ctrl+C/체크포인트 경로를 SemiSup 까지
확장하고, 잘못된 트레이너 타입으로 resume 하는 실수를 방지.

**변경**
- `src/trainer/checkpoint.rs`
  - `pub enum ParadigmTag { Supervised, Unsupervised, SemiSupervised,
    Autoregressive, Reinforcement }` + `as_str()`.
  - `TrainingCheckpoint` 에 `paradigm: Option<ParadigmTag>` +
    `rng_seed: u64` 필드 추가 (`#[serde(default)]` 로 레거시 JSON 호환).
  - `TrainingCheckpoint::verify_paradigm(&self, expected) -> MlResult<()>` —
    `None` 은 통과(legacy), 불일치 시 Err.
  - `save_interrupt_checkpoint<F: FnOnce(&str) -> MlResult<()>>` 헬퍼 —
    4 트레이너에 중복되던 45라인 save 로직을 한 곳으로.
- `src/trainer/supervised.rs`, `unsupervised.rs`, `autoregressive.rs` —
  인터럽트 시 `save_interrupt_checkpoint` 호출, `resume` 에
  `verify_paradigm(…)?` 삽입.
- `src/trainer/semi_supervised.rs` — 기존 `fit` 단일 메서드를
  `fit`/`resume`/`fit_inner` 로 분해. Ctrl+C 감지 + save 경로 신규 추가.
- `src/trainer/checkpoint.rs::tests` — 왕복/paradigm mismatch/legacy
  load 테스트 4건.

**테스트**: 185 passed (+4).

**미완** (`TRAINER_NEXT_PHASES.md::Phase 4.1 / 6 / 7` 으로 분리)
- 옵티마이저 내부 상태(Adam m/v 등) 직렬화 → **Phase 6**.
- RL 체크포인트 → **Phase 7**.
- 트레이너 레벨 save→resume 왕복 테스트 → **Phase 4.1**.

### Phase 3.1 — 훅 경로 일원화 (완료)

**목적**: Phase 3 에서 훅 인프라만 활성화되고 프리셋은 여전히 `*EpochStep`
내부 인라인 경로로 accuracy/PPL 를 계산하던 이중 경로를 단일화.

**변경**
- `src/trainer/supervised.rs`
  - `From<Trainer> for SupervisedTrainer` 가 `cfg.metrics.accuracy` 참일 때
    `ClassificationAccuracy` 훅을 자동 장착.
  - `SupervisedEpochStep` 에서 `accuracy` / `accuracy_enabled` 필드 제거,
    인라인 `accuracy.update(…)` 및 `format_epoch_extras` 의 `"AC: ..."` 포맷 제거.
  - 훅 자동 장착 단위 테스트 7건 추가.
- `src/trainer/autoregressive.rs`
  - `From<Trainer> for AutoregressiveTrainer` 가 `cfg.epoch_log_interval`
    이 유효한 경우 (silent 제외) `Perplexity` 훅을 자동 장착.
  - `AutoregressiveEpochStep` 에서 `ppl` 필드와 배치/에폭 인라인 `"PPL: ..."`
    포매팅 제거 — 훅 경로로 일원화.
  - 훅 자동 장착 단위 테스트 6건 추가.
- 프리셋 계약:
  | Preset   | Supervised 훅 | Autoregressive 훅 |
  |----------|---------------|-------------------|
  | silent   | ✗ (perf)      | ✗ (perf)          |
  | minimal  | ✓ accuracy    | ✓ perplexity      |
  | default  | ✓ accuracy    | ✓ perplexity      |
  | verbose  | ✓ accuracy    | ✓ perplexity      |
- `from_config` 원시 경로는 자동 훅 대상이 아니며, 사용자가 필요 시
  `.with_hook(…)` / `.with_perplexity()` 로 직접 붙인다.

**테스트**: 185 → 198 passed (+13).

**로그 포맷 영향**
- 에폭 요약의 `"AC: …"` / `"PPL: …"` 는 위치·포맷이 동일하게 유지됨 (훅 출력 경로).
- AR 의 **배치 레벨 live PPL** 은 배치 로그에서 제거됨 (에폭 요약에만 남음).
  이는 로그 노이즈 감소 목적의 의도적 변경.

### Phase 5 — SupervisedTrainer 승격 + 패러다임별 빌더 (부분 완료)

**목적**: `SupervisedTrainer` 가 `pub use Trainer as …` 단순 별칭에
머물러 있어 다른 패러다임 트레이너와 타입 구조가 비대칭이었다. 실제
지도학습 루프를 전담하는 별도 타입으로 승격.

**변경**
- `src/trainer/supervised.rs` — `pub use` 제거, `pub struct
  SupervisedTrainer { core: TrainerCore }` 로 전환. `From<Trainer>`,
  `from_config/from_core`, presets(`silent/minimal/default/verbose`),
  `builder()` 제공. `fit/resume/fit_inner` + `SupervisedEpochStep` 을
  `mod.rs` 로부터 이동 (약 350라인).
- `src/trainer/mod.rs` — `Trainer` 를 순수 팩토리로 축소 (presets +
  builder 만). 자체 `fit` 는 제거됨. 문서 주석도 갱신.
- **패러다임별 `with_hook`**: 5 트레이너 각각에
  `with_hook(Box<dyn MetricHook>) -> Self` 추가 (체이닝 스타일).
- `src/trainer/autoregressive.rs` — `with_perplexity()` 편의 메서드.
- `SemiSupervisedTrainer::with_ramp`, `RLTrainer::with_gamma/with_baseline`
  는 Phase 5 이전부터 존재.

**설계 변경** (원 플랜 대비)
- 원 계획의 `SupervisedBuilder`/`AutoregressiveBuilder` 같은 **별도 빌더
  struct** 는 도입하지 않았다. 대신 trainer 자체에 `with_*` 메서드로
  knob 을 달았다. 최종 UX 동등하며 타입 수 절감.

**테스트**: 185 passed (회귀 없음).

**미완** (`TRAINER_NEXT_PHASES.md::Phase 3.1` 으로 분리)
- `default()` 의 자동 훅 장착.
- `StepUnit { Batch, Episode, Token }` enum — cosmetic, 우선순위 낮음.

---

## 2. 현재 코드베이스 스냅샷

```
src/trainer/
├── mod.rs                  (Trainer — 순수 팩토리: presets + builder)
├── supervised.rs           (SupervisedModel trait + SupervisedTrainer + EpochStep)
├── autoregressive.rs       (AutoregressiveTrainer + AutoregressiveEpochStep + with_perplexity)
├── unsupervised.rs         (UnsupervisedTrainer + UnsupervisedEpochStep)
├── semi_supervised.rs      (SemiSupervisedTrainer + SemiSupervisedEpochStep + Ctrl+C/resume)
├── reinforcement.rs        (RLTrainer — Phase 7 에서 체크포인트 추가 예정)
├── checkpoint.rs           (TrainingCheckpoint, ParadigmTag, save_interrupt_checkpoint, Ctrl+C)
├── progress.rs             (indicatif 래퍼)
└── core/
    ├── mod.rs              (TrainerCore — hooks: RefCell<Vec<Box<dyn MetricHook>>>)
    ├── config.rs           (LogConfig, Metrics, TrainerBuilder)
    ├── convergence.rs      (Convergence)
    ├── epoch_loop.rs       (EpochStep trait + last_pred/target/n_tokens/lambda 접근자)
    ├── metric_hook.rs      (MetricHook, BatchContext — 배치마다 update 호출됨)
    └── metrics.rs          (grad_norm, accuracy, Perplexity + MetricHook impl)
```

---

## 3. 진행 예정 작업

Phase 3/4/5 의 scope 제한 미이행 항목, 옵티마이저 상태 직렬화, RL
체크포인트, loss 감사는 **`TRAINER_NEXT_PHASES.md`** 로 이관되어 다음
페이즈로 추적된다:

| Phase | 내용                                           | 선행 의존    | 상태   |
|-------|------------------------------------------------|--------------|--------|
| 8     | 플랜/보고서 문서 동기화                        | 즉시         | ✅ done |
| 3.1   | 훅 경로 일원화 (프리셋 자동 장착)              | 없음         | ✅ done |
| 4.1   | 체크포인트 save→resume 왕복 테스트             | 없음         | 대기   |
| 6     | Optimizer snapshot/restore (6 impl + trait)    | 4.1 선행 권장 | 대기   |
| 7     | RL Ctrl+C + 체크포인트                         | 6 선행 권장  | 대기   |
| B1    | 기타 loss per-row 버그 감사                    | 없음         | 대기   |

상세 설계는 `TRAINER_NEXT_PHASES.md` 참조.

---

## 4. 기타 후속 이슈 (차기 페이즈 후보)

- **다른 loss 들의 per-row 버그 여부**: `CrossEntropyLoss`,
  `BinaryCrossEntropyLoss`, `MSE`, `MAE`, `Huber` 에 대해 B>1 입력으로
  스모크 테스트 하고 동일 패턴이 있으면 수정.
- **실 언어모델 파일럿**: Bigram 이외에 Transformer decoder + AR trainer
  조합 테스트. `src/tests/common/model/transformer/` 신설.
- **LatentDiffusion 정식 구현**: Encoder/Decoder/Scheduler 래퍼를 활용해
  VAE 기반 latent-space DDPM 완성.
- **로그 포맷 회귀 감시**: Phase 2 에서 SemiSup `λ` 가 배치 로그 뒤로
  이동한 사례 — 전체 포맷을 골든 파일로 고정할지 여부 결정.

---

## 5. 체크 포인트 명령어

```bash
# 전체 테스트
cargo test --lib --features enableBackward

# 특정 페이즈 영향 확인
cargo test --lib --features enableBackward autoregressive
cargo test --lib --features enableBackward semi_supervised
cargo test --lib --features enableBackward softmax_ce

# 컴파일 위생 (0 errors 기대)
cargo check --lib --features enableBackward
```
