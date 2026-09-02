# Trainer Remediation Plan

원 작업 요청("자기회귀 모델 전용 트레이너 구현 + 다양한 아키텍처 지원")의
보완 계획. 페이즈 순서대로 단계적으로 실행한다.

최종 수정일: 2026-04-22 (Phase 3.1 완료 반영)

> **후속 계획**: Phase 3/4/5 에서 scope 제한으로 미이행된 항목과 Optimizer
> 상태 직렬화, RL 체크포인트, loss 감사는 **별도 문서
> `TRAINER_NEXT_PHASES.md`** 로 분리되어 Phase 6~8/B1 로 추적된다.

---

## 전체 상태

| Phase | 내용                                               | 상태          |
|-------|----------------------------------------------------|---------------|
| 0     | 스텁 보존 + 컴파일 위생                            | ✅ done       |
| 1     | `AutoregressiveTrainer` + `AutoregressiveModel`     | ✅ done       |
| B0    | `SoftmaxCrossEntropyLoss` per-row 버그 수정        | ✅ done       |
| 2     | `TrainerCore::run_epoch` 공통 루프 추출            | ✅ done       |
| 3     | `MetricHook` 플러그인 활성화                       | ✅ done (코어) |
| 4     | 체크포인트/인터럽트 통합                           | ⚠️ partial    |
| 5     | `SupervisedTrainer` 승격 + 패러다임별 빌더         | ⚠️ partial    |
| 3.1   | 훅 경로 일원화 (프리셋 자동 장착)                  | ✅ done       |

**테스트**: 177 (Phase 2) → 181 (Phase 3) → 185 (Phase 4/5) → 198 (Phase 3.1) passed.

Phase 4/5 의 나머지 미완 항목은 `TRAINER_NEXT_PHASES.md` 의 Phase 4.1 /
6 / 7 에 포워딩되어 있다.

---

## 사용자 불변 지침

- **LatentDiffusion / Encoder / Decoder / Scheduler 래퍼는 제거 금지.**
  `LatentDiffusion` 구현 시 재사용 예정. 현재는 `#[allow(dead_code)]` + TODO
  주석으로 보존되어 있다.
- 페이즈는 순서대로 실행하고, 각 페이즈 종료 시 평가 후 다음으로.
- 작업 중 발견한 버그는 별도 B-페이즈(`B0`, `B1` 등)로 기록하고 기회가
  생길 때 처리한다.

---

## Phase 0 — 스텁 보존 + 컴파일 위생 (완료)

**수행 내용**
- `src/tests/common/model/diffusion/decoder.rs`, `encoder.rs::Encoder`,
  `scheduler.rs::Scheduler`, `mod.rs::LatentDiffusion` — `#[allow(dead_code)]`
  + TODO 주석 추가.
- `diffusion/mod.rs` 의 dead `use` (Decoder, Encoder, Scheduler) 제거.
- `cargo check --tests` 통과, 신규 경고 없음.

---

## Phase 1 — AutoregressiveTrainer (완료)

**수행 내용**
- 신규 파일:
  - `src/trainer/autoregressive.rs` — `AutoregressiveModel` 트레잇 +
    `AutoregressiveTrainer` 구조체. `forward_loss(x) → (logits, loss, n_tokens)`
    계약. 체크포인트 / Ctrl+C / PPL 에폭 로그 지원.
  - `src/tests/common/model/autoregressive/mod.rs` — Bigram LM 파일럿.
- 수정:
  - `src/trainer/core/metrics.rs` — `Perplexity` 누적기 + `MetricHook` impl.
  - `src/trainer/core/mod.rs`, `src/trainer/mod.rs` — 재수출 갱신.
  - `src/tests/common/model/mod.rs` — `pub mod autoregressive`.
- 테스트: 40/40 model 테스트 통과.

---

## Phase B0 — SoftmaxCrossEntropyLoss per-row 버그 수정 (다음 작업)

### 문제

`src/loss/function.rs::SoftmaxCrossEntropyLoss::{forward, backward}` 가
`[B, V]` 입력에서 **전체 플랫 텐서를 하나의 분포**로 보고 log-sum-exp 를
계산한다. B=1 경로에서는 드러나지 않는 잠복 버그.

증상: B>1 (AR 파일럿의 `[L, V]` 시퀀스) 에서 CE loss 가 음수가 될 수 있음.
실제로 Phase 1 에서 AR 파일럿이 `-5.68` 을 반환해 발견.

### 해결 계획

1. `forward` 를 per-row log-sum-exp 로 수정:
   - shape `[B, V]` 를 B 행으로 분리해 각 행의 `(max + log_sum_exp) - dot(z, t)` 계산.
   - `scalar!(Σ_batch / B)` 로 평균 반환.
2. `backward` 를 동일한 per-row softmax 로 수정:
   - B 행마다 softmax, `(p - t)` 계산 후 concat.
3. 회귀 테스트:
   - 기존 `[1, V]` 케이스가 계속 통과해야 함 (softmax regression, MLP 등).
   - B>1 케이스에 대한 단위 테스트 추가 (loss ≥ 0, 수치 정답과 일치).
4. AR 파일럿의 per-row 루프 우회 코드(`BigramLM::forward_loss` 의 t 루프)는
   **그대로 유지**한다 — AR 의 teacher-forcing 시맨틱을 명시적으로 드러내는
   데 유용하며, 성능 최적화(batched)는 Phase 2 이후에 생각한다.
   - 단, 버그 수정 후 `[L, V]` 한 번에 loss 를 계산해도 결과가 같음을
     확인하는 보조 테스트를 둔다.

### 범위 제한

- 다른 loss (`CrossEntropyLoss`, `MSE`, `MAE`, `Huber`, `BCE`) 의 동일
  패턴 버그는 B0 범위 밖. 별도 버그 페이즈로 기록.

---

## Phase 2 — TrainerCore::run_epoch 공통 루프 추출 (완료)

**수행 내용**
- 신규 파일: `src/trainer/core/epoch_loop.rs` — `EpochStep` trait, `StepInfo`,
  `EpochOutcome`, `TrainerCore::run_epoch<S: EpochStep>`. 배치 바 관리, NaN
  감지, 배치 로그 포매팅, optimizer.step, 인터럽트 감지를 공통화.
- `Trainer::fit_inner` (지도) → `SupervisedEpochStep` 어댑터.
  `ClassificationAccuracy` 누적을 step 내부로 이동.
- `UnsupervisedTrainer::fit_inner` → `UnsupervisedEpochStep`.
- `AutoregressiveTrainer::fit_inner` → `AutoregressiveEpochStep`.
  `Perplexity` 누적을 step 내부로 이동; 배치/에폭 로그에 `PPL: ...` 유지.
- `SemiSupervisedTrainer::fit` → `SemiSupervisedEpochStep`. `lambda` 는
  에폭당 고정이므로 step 생성 시 주입. 인터럽트/체크포인트는 Phase 4 에서
  통합 예정 (현재 `run_epoch` 호출 시 `interrupt: None` 으로 유지).
- RL 은 에피소드 시맨틱이라 범위 외로 유지.
- 테스트: 177/177 전체 통과.

### 후속 사항

- SemiSup 의 `λ` 와 AR 의 `PPL` 이 `extra_msg` 에 들어가는 구조 상, 배치
  로그 포맷이 원본 `FW/BW | λ | GN | UR` → 신규 `FW/BW | GN | UR | λ` 로
  뒤집혔다. 기능 변화는 없지만 로그 회귀 감시 시 인지 필요.

---

## Phase 2 (설계 메모, 참고용)

### 동기

`Trainer::fit_inner` (지도), `UnsupervisedTrainer::fit_inner`,
`SemiSupervisedTrainer::fit_inner`, `AutoregressiveTrainer::fit_inner`
네 곳에 **거의 동일한 에폭 루프** 가 반복된다. 총 ~900 라인의 중복.

### 설계

```rust
impl TrainerCore {
    pub fn run_epoch<F, R>(
        &self,
        optimizer: &mut dyn Optimizer,
        n_samples: usize,
        epoch_idx: usize,
        epoch_bounds: (usize, usize),        // (start, total)
        progress: &EpochProgress,
        interrupt: Option<&InterruptFlag>,
        mut step: F,
    ) -> MlResult<EpochOutcome>
    where
        F: FnMut(&mut dyn Optimizer, usize /* batch_idx */) -> MlResult<StepResult>,
    {
        // 배치 바 생성 → 인덱스 루프 → NaN 검사 → 메트릭 로그
        // → optimizer.step → 인터럽트 감지 → 에폭 요약
    }
}

pub struct StepResult {
    pub loss: f32,
    pub params: Vec<*const dyn Parameter>, // or similar lifetime-elided form
    pub extra: StepExtra,                  // timing, token count, pred/target refs
}

pub struct EpochOutcome {
    pub avg_loss: f32,
    pub interrupted: bool,
    pub hooks_snapshot: Vec<f32>,
}
```

### 범위

- **지도/비지도/반지도/AR** 네 트레이너가 `run_epoch` 를 호출하도록 이전.
- **RL 은 범위 외** — 배치 루프가 아닌 에피소드 루프 시맨틱이다.
  별도의 `run_episode<F>` 헬퍼를 추가할지는 Phase 2 말미에 재평가.
- 인터럽트/체크포인트 저장 로직도 `run_epoch` 로 끌어올린다.
- 마이그레이션은 파라다임별 단계:
  1. Supervised → `run_epoch` 기반으로 재작성, 기존 테스트 통과 확인.
  2. Unsupervised 동일.
  3. SemiSupervised (λ 램프는 closure 내부에서 처리).
  4. Autoregressive (PPL 누적은 closure 외부의 공유 상태로).

### 리스크

- 클로저 캡처로 `model: &mut M` 을 넣어야 하는데, 다른 호출 지점에서
  `model.params()` 를 동시에 참조하려 하면 borrow checker 충돌 가능.
- 해결 전략: `params()` 를 클로저 내부에서만 호출하고 결과(f32)만 밖으로
  내보낸다. 또는 `StepResult::post_step_params<'a, 'b>(&'b self) -> &'b [...]`
  로 참조 생명주기를 명시.

---

## Phase 3 — MetricHook 플러그인 활성화 (코어 완료, 마무리는 3.1)

### 실제 구현 결과

- `TrainerCore::hooks` 를 `RefCell<Vec<Box<dyn MetricHook>>>` 로 변환
  (`&self` 체이닝 지원).
- `EpochStep` trait 에 `current_lr` / `last_pred` / `last_target` /
  `last_n_tokens` / `last_lambda` 접근자 추가 — 각 `*EpochStep` 이 배치별
  관측치를 `Option<Variable>` 필드에 stash.
- `run_epoch` 가 매 배치마다 `BatchContext` 조립 후 `hook.update(&ctx)`
  호출, 에폭 종료 시 `hook.format()` 을 `summary_extras` 에 합류.
- 훅 미장착 시 `hooks.borrow().is_empty()` 가드로 zero-overhead 경로 보존.
- `ClassificationAccuracy`, `Perplexity` 의 `MetricHook` impl 갱신
  + 단위 테스트 3건 추가 → 181 passed.

### 미이행 → Phase 3.1 (NEXT_PHASES 문서)

- `default()` / `verbose()` 프리셋의 **자동 훅 장착** 미이행.
- 인라인 경로(`*EpochStep::accuracy`, `ppl` 필드)는 여전히 공존.
- "훅 경로 ↔ 인라인 경로 bit-identical" 골든 테스트 미추가.

이상 3건은 `TRAINER_NEXT_PHASES.md` 의 **Phase 3.1** 에서 처리.

---

## Phase 4 — 체크포인트/인터럽트 통합 (부분 완료)

### 실제 구현 결과

| 트레이너          | Ctrl+C | 체크포인트 | `resume()` | Optim m/v |
|-------------------|--------|-----------|-----------|-----------|
| Supervised        | ✅     | ✅        | ✅        | ❌         |
| Unsupervised      | ✅     | ✅        | ✅        | ❌         |
| SemiSupervised    | ✅ *   | ✅ *      | ✅ *      | ❌         |
| Reinforcement     | ❌     | ❌        | ❌        | ❌         |
| Autoregressive    | ✅     | ✅        | ✅        | ❌         |

`*` Phase 4 에서 신규 추가된 경로.

**신규**
- `ParadigmTag` enum + `TrainingCheckpoint.paradigm: Option<ParadigmTag>`
  (`#[serde(default)]` 로 레거시 체크포인트 호환).
- `TrainingCheckpoint.verify_paradigm(expected)` — 잘못된 트레이너 타입
  재개 방지.
- `TrainingCheckpoint.rng_seed: u64` 필드 (`#[serde(default)]`).
- `save_interrupt_checkpoint` 헬퍼로 4곳 중복 save 로직 제거.
- SemiSupervisedTrainer 의 `fit` → `fit`/`resume`/`fit_inner` 분해
  + Ctrl+C 경로 신규.
- 체크포인트 단위 테스트 4건 추가 → 185 passed.

### 미이행 → NEXT_PHASES

- **옵티마이저 내부 상태(Adam m/v 등) 직렬화** — `Optimizer` trait 확장이
  필요해 **`TRAINER_NEXT_PHASES.md::Phase 6`** 로 분리.
- **RL 의 Ctrl+C/체크포인트** — 에피소드 루프 시맨틱이라 **Phase 7** 로 분리.
- **각 패러다임 save→resume 왕복 테스트** — **Phase 4.1** 에서 추가.

---

## Phase 5 — SupervisedTrainer 승격 + 패러다임별 빌더 (부분 완료)

### 실제 구현 결과

- `SupervisedTrainer` 를 `pub struct SupervisedTrainer { core: TrainerCore }`
  로 승격. `fit/resume/fit_inner` + `SupervisedEpochStep` 을
  `src/trainer/mod.rs` → `src/trainer/supervised.rs` 로 이동.
- `From<Trainer>` 구현으로 기존 `Trainer::default().into()` 체이닝 유지.
- `Trainer` 는 **순수 팩토리**(presets + builder) 로 역할 축소 — 자체
  `fit` 는 더 이상 없음.
- 모든 패러다임에 `with_hook(Box<dyn MetricHook>)` 추가.
- `AutoregressiveTrainer::with_perplexity()` 편의 메서드 추가.
- 185/185 유지.

### 설계 변경 (원 계획 대비)

원 계획은 `SupervisedBuilder`/`AutoregressiveBuilder` 등 **별도 빌더 struct**
도입이었으나, 실제로는 각 트레이너 자체에 `with_*` 메서드(`with_gamma`,
`with_baseline`, `with_ramp`, `with_hook`, `with_perplexity`)로 달렸다.
최종 UX 는 동등하나 타입 구조가 계획과 다름 — 허용 가능한 설계 변경으로 확정.

### 미이행 → NEXT_PHASES

- **`StepUnit { Batch, Episode, Token }` enum** — cosmetic 로그 포맷
  통일. 우선순위 낮음. `TRAINER_NEXT_PHASES.md` 추후 후보.
- **`default()` 의 자동 훅 장착** — Phase 3.1 과 묶임.

---

## 향후 발견될 수 있는 후속 이슈 (차기 페이즈 후보)

- **다른 loss 들의 per-row 버그 여부** 확인 (`CrossEntropyLoss`, `BCE` 등).
- **실 언어모델 파일럿** 추가 — 지금은 bigram 뿐. `src/tests/common/model/
  transformer/` 에 실제 Transformer decoder + AR trainer 조합 테스트.
- **LatentDiffusion 구현** (사용자 지정 보존 스텁들의 정식 구현).
