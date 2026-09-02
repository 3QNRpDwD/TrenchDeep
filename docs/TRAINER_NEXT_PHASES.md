# Trainer Remediation — 후속 작업 상세 계획서

> API 기반 리팩터링은 2026-09-01 완료되었다. Phase 6/7/B1은 [`TRAINER_API_OPTIMIZATION.md`](TRAINER_API_OPTIMIZATION.md)의 확정 계약을 따른다.

**작성일**: 2026-04-20
**대상 브랜치**: `V4`
**선행 문서**: `TRAINER_REMEDIATION_PLAN.md`, `TRAINER_PROGRESS_REPORT.md`
**테스트 기준**: `cargo test --lib --features enableBackward` 185/185 passed

---

## 0. 배경

원 `TRAINER_REMEDIATION_PLAN.md` 의 Phase 0~5 실행 후 평가에서 다음 사실이 드러남:

1. Phase 3~5 는 **기본 구조는 이행되었으나 일부 세부 항목이 scope 제한으로 남음**.
   - 프리셋 자동 훅 장착 미이행
   - 골든 테스트(훅/인라인 bit-identical) 미추가
   - 체크포인트 왕복 테스트 미추가
2. Phase 4 는 **옵티마이저 상태 직렬화**(Adam m/v 등)가 Optimizer trait 확장을
   요구해 독립 페이즈로 분리됨.
3. RL 트레이너는 배치 루프가 아닌 에피소드 루프라 **체크포인트/Ctrl+C 경로가
   아직 없음**.
4. `SoftmaxCrossEntropyLoss` 외 다른 손실들의 per-row 버그 여부는 미확인.
5. 기존 플랜/보고서 문서는 **Phase 2 종료 상태에 정지**되어 있어 stale.

본 문서는 위 항목들을 묶어 Phase 6 ~ Phase B1 까지의 실행 계획을 제시한다.

---

## 1. 전체 페이즈 요약

| Phase | 내용                                             | 난이도 | 우선순위 | 의존성        |
|-------|--------------------------------------------------|--------|----------|---------------|
| 3.1   | 훅 경로 일원화 (프리셋 자동 장착 + 골든 테스트)  | S      | 중       | 없음          |
| 4.1   | 각 패러다임 체크포인트 왕복 테스트               | S      | 중       | 없음          |
| 6     | Optimizer snapshot/restore (trait 확장 + 6 impl) | L      | 높음     | 4.1 선행 권장 |
| 7     | RL 트레이너 Ctrl+C + 체크포인트                  | M      | 중       | 6 선행 권장   |
| B1    | 기타 loss 의 per-row 버그 감사                   | S~M    | 중       | 없음          |
| 8     | 플랜/보고서 문서 동기화                          | XS     | 즉시     | 없음          |

**난이도 기준**: XS(<1h) / S(1-3h) / M(3-8h) / L(8h+)

---

## Phase 8 — 플랜/보고서 문서 동기화 (즉시 실행)

### 목적

`TRAINER_REMEDIATION_PLAN.md` 와 `TRAINER_PROGRESS_REPORT.md` 가 Phase 2 종료
상태에 정지되어 있어, 신규 참여자가 현재 진행 상황을 잘못 파악할 위험.

### 변경

1. **`TRAINER_REMEDIATION_PLAN.md`**
   - 상태표(§전체 상태)에서 Phase 3/4/5 를 `✅ done` 으로, B0 옆에 체크 추가
     (이미 표시되어 있음).
   - 각 페이즈 본문에 "실제 구현 결과 / 미이행 항목" 짧은 섹션 추가.
   - Phase 3/4/5 의 미이행 항목은 본 문서의 Phase 3.1/4.1/6/7 로 포워딩.
2. **`TRAINER_PROGRESS_REPORT.md`**
   - §1 에 Phase 3/4/5 완료 절 추가 (파일 변경, 테스트 수, 주요 결정).
   - §2 스냅샷을 현재 구조로 갱신 (`SupervisedTrainer` 승격, `supervised.rs` 가
     실제 루프를 가짐).
   - §3 "진행 예정 작업"을 본 문서 참조로 교체.

### 검증

- `docs/TRAINER_REMEDIATION_PLAN.md` 상태표의 Phase 와 `Cargo.toml` 브랜치
  head 의 실제 코드 위치가 일치.
- `docs/TRAINER_PROGRESS_REPORT.md` 의 테스트 개수(185)가 `cargo test --lib
  --features enableBackward` 결과와 일치.

---

## Phase 3.1 — 훅 경로 일원화

### 목적

Phase 3 에서 `MetricHook` 플러그인을 활성화했지만, 기본 accuracy/PPL 계산은
여전히 `*EpochStep` 내부의 인라인 경로를 사용한다. 두 경로가 공존해 **의미
동일한 메트릭이 서로 다른 코드에서 중복**으로 계산된다. 훅 경로로 일원화해
중복을 제거하고, Phase 3 계획의 "bit-identical 골든 테스트" 로 회귀를 고정.

### 현재 상태

- `run_epoch` 는 `hooks_active` 검사 후 배치마다 `hook.update(&ctx)` 호출.
- `SupervisedEpochStep::accuracy` 필드와 `AutoregressiveEpochStep::ppl` 필드는
  **여전히 인라인으로** 계산·포맷한다.
- 프리셋(`*::default()/silent()/…`)은 훅을 자동 장착하지 않음. 사용자가
  `with_hook`/`with_perplexity` 를 명시적으로 호출해야 함.

### 할 일

1. **프리셋 자동 장착**
   - `SupervisedTrainer::default()` / `::verbose()` → 자동으로
     `ClassificationAccuracy` 훅 장착.
   - `AutoregressiveTrainer::default()` / `::verbose()` → `Perplexity` 훅.
   - `silent()` / `minimal()` 은 추가 훅 없이 유지.

2. **인라인 경로 deprecation 전환**
   - `SupervisedEpochStep::accuracy{,_enabled}` 필드 제거하고
     `format_epoch_extras` 에서 훅 경로 결과를 인용하도록 변경.
     → 단, 훅이 없는 경우의 fallback 은 어떻게? 두 가지 옵션:
       - (A) **항상 훅 장착** 원칙. `LogConfig.metrics.accuracy=true` 이면
         빌더가 자동으로 훅 추가. → `default()` 가 이미 accuracy 를 켜므로
         자연스러움.
       - (B) accuracy 계산은 유지하되, `format_epoch_extras` 만 훅 미존재
         시에만 인라인 값을 사용. → 과도기 안전책.
   - 1차 구현은 (B) 로 안전하게, 2차로 (A) 로 정리 권장.

3. **Bit-identical 골든 테스트 (Phase 3 계획 항목)**
   - 고정 시드 + 소규모 코퍼스로 **인라인 경로**와 **훅 경로**의 값이 동일한지
     확인. `ClassificationAccuracy` 와 `Perplexity` 각각에 대해 1건씩.
   - `src/trainer/core/metrics.rs::tests` 에 추가.

### 변경 파일

- `src/trainer/supervised.rs` — `default/verbose` 에 `.with_hook(Box::new(ClassificationAccuracy::new()))`.
- `src/trainer/autoregressive.rs` — 동일 요령으로 `Perplexity`.
- `src/trainer/core/metrics.rs` — 골든 테스트.
- 선택적으로 `*EpochStep::format_epoch_extras` 수정 (옵션 B).

### 검증

- 기존 185/185 유지, 신규 골든 테스트 2~3건 추가 → 187~188 passed 예상.
- `default()` 사용한 파일럿 테스트들(`mnist_test`, `softmax` 등) 의 로그
  포맷 회귀 없음.

### 리스크 / 범위 외

- 기존 mnist/AR 파일럿이 `default()` 를 쓰면서 로그 문자열에 의존하는
  assertion 이 있으면 깨질 수 있음. **grep 으로 사전 확인** 필요.
- `LogConfig.metrics` 비트플래그 완전 제거는 본 페이즈 범위 외.

---

## Phase 4.1 — 체크포인트 왕복 테스트

### 목적

Phase 4 에서 `ParadigmTag` + `save_interrupt_checkpoint` 헬퍼를 도입했지만,
"save → reload → resume → 동일 결과" 를 검증하는 통합 테스트가 없음. 헬퍼의
행위적 정확성을 고정.

### 현재 상태

- `src/trainer/checkpoint.rs::tests` 에 직렬화 왕복/paradigm mismatch 등
  단위 테스트 4건 추가됨.
- **트레이너 레벨**에서 모델·옵티마이저 상태를 save 하고 같은 모델을
  resume 해 산출 로스를 비교하는 테스트는 없음.

### 할 일

1. 각 패러다임에 대해 "save → kill → resume → 수렴 손실 동일" 테스트:
   - Supervised: softmax regression 소형 데이터셋, 10 epoch 완주 vs
     5 epoch save + 5 epoch resume → 최종 loss 가 허용 오차 내 동일해야.
   - Unsupervised: AE 토이 예제로 동일 패턴.
   - Autoregressive: BigramLM 으로 동일 패턴.
   - SemiSupervised: Phase 4 에서 추가된 resume 경로 검증 (소형 MLP).
2. **옵티마이저 상태가 초기화되는 현재 한계**를 테스트에 주석으로 명시
   — "Adam m/v 재시작으로 resume 후 수 에폭은 재수렴 흔적이 있을 수 있음".
3. 각 테스트는 `tempfile` 크레이트로 임시 디렉토리 사용, 테스트 후 정리.

### 변경 파일

- `src/tests/common/model/nonlinear/softmax.rs` (또는 신규
  `src/tests/common/model/supervised_checkpoint.rs`).
- 각 패러다임 테스트 모듈에 유사 테스트 추가.
- `Cargo.toml` — `tempfile` dev-dependency (이미 있다면 skip).

### 검증

- 4건 추가로 189 passed.
- 각 왕복 테스트는 `tolerance = loss_diff < 5%` 등 느슨한 기준으로.
  옵티마이저 상태가 복원되지 않으므로 bit-identical 은 기대 불가.

### 리스크 / 범위 외

- Adam m/v 미저장으로 인한 loss 차이가 5% 초과 시 테스트 실패 → Phase 6
  완료 후 tolerance 를 bit-identical 수준으로 조일 수 있음.
- Supervised 의 왕복 테스트는 Phase 4.1 로 즉시 가능, 옵티마이저 상태
  로 인한 회귀는 Phase 6 시점에 재평가.

---

## Phase 6 — Optimizer snapshot/restore (핵심)

### 목적

현재 체크포인트는 `optimizer_lr` 만 저장한다. Adam 의 `m/v`, Momentum 의
`velocity` 등 **옵티마이저 내부 상태**가 resume 시 초기화되어, 중단 시점의
유효 학습률이 사라진다. 긴 학습에서 resume 후 수 에폭 동안 재수렴이
필요해 실질적 학습 손실이 발생.

### 현재 구현 현황

6 개 옵티마이저 존재 (`src/optimizer/`):

| 옵티마이저 | 내부 상태 (snapshot 대상)                        |
|------------|--------------------------------------------------|
| SGD        | 없음 (stateless)                                 |
| Momentum   | `velocity: Vec<Vec<f32>>`                        |
| AdaGrad    | `sum_sq_grad: Vec<Vec<f32>>`                     |
| RMSProp    | `ema_sq_grad: Vec<Vec<f32>>`                     |
| Adam       | `m: Vec<Vec<f32>>`, `v: Vec<Vec<f32>>`, `t: usize` |
| AdamW      | Adam 과 동일 + `weight_decay: f32`               |

모두 `Vec<Vec<f32>>` 버퍼 + 스칼라 몇 개로 직렬화 가능. 외부 블롭 없음.

### 설계

#### 6.1 `Optimizer` trait 확장

```rust
pub trait Optimizer {
    // 기존 메서드 유지 …
    fn step(&mut self) -> MlResult<()>;
    fn zero_grad(&self) -> MlResult<()>;
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);

    /// 옵티마이저 내부 상태를 직렬화 가능한 형태로 스냅샷.
    /// stateless 옵티마이저(SGD)는 빈 스냅샷을 반환.
    fn snapshot(&self) -> OptimizerSnapshot;

    /// 스냅샷으로부터 내부 상태를 복원.
    /// 옵티마이저 타입이 일치하지 않으면 `Err`.
    fn restore(&mut self, snapshot: &OptimizerSnapshot) -> MlResult<()>;

    /// 자기 옵티마이저의 태그 (타입 안전성 검사용).
    fn kind(&self) -> OptimizerKind;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerKind { SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerSnapshot {
    pub kind:    OptimizerKind,
    pub lr:      f32,
    pub buffers: Vec<Vec<f32>>,   // 옵티마이저별 해석 (아래 표 참조)
    pub scalars: Vec<f32>,        // beta1, beta2, t_step 등
}
```

**버퍼 레이아웃 (kind 별)**:

| Kind     | `buffers`                    | `scalars`                         |
|----------|------------------------------|-----------------------------------|
| SGD      | `[]`                         | `[]`                              |
| Momentum | `[velocity_i for i in params]` | `[momentum_coef]`                |
| AdaGrad  | `[sum_sq_i]`                 | `[eps]`                           |
| RMSProp  | `[ema_sq_i]`                 | `[alpha, eps]`                    |
| Adam     | `[m_i…] + [v_i…]`            | `[beta1, beta2, eps, t as f32]`   |
| AdamW    | `[m_i…] + [v_i…]`            | `[beta1, beta2, eps, wd, t as f32]` |

#### 6.2 `TrainingCheckpoint` 확장

```rust
pub struct TrainingCheckpoint {
    // 기존 필드 …
    pub optimizer_lr: f32,
    pub paradigm: Option<ParadigmTag>,
    pub rng_seed: u64,

    /// 옵티마이저 내부 상태 스냅샷. 레거시 체크포인트(없음) 호환.
    #[serde(default)]
    pub optimizer_snapshot: Option<OptimizerSnapshot>,
}
```

- `save_interrupt_checkpoint` 헬퍼 시그니처에 `&dyn Optimizer` 를 추가해
  `opt.snapshot()` 호출.
- `resume` 경로에서 `if let Some(snap) = &ckpt.optimizer_snapshot {
  opt.restore(snap)?; }`.

### 할 일

1. `src/optimizer/mod.rs` — trait 에 `snapshot/restore/kind` 추가,
   `OptimizerSnapshot` / `OptimizerKind` 정의. `serde` 의존성 이미 있음.
2. 각 옵티마이저 파일에 `snapshot/restore/kind` 구현 (6 파일).
3. `src/trainer/checkpoint.rs`
   - `TrainingCheckpoint` 에 `optimizer_snapshot` 필드 추가
     (`#[serde(default)]`).
   - `save_interrupt_checkpoint` 시그니처에 optimizer 추가.
4. 각 패러다임 트레이너 (`supervised`, `unsupervised`, `autoregressive`,
   `semi_supervised`) 의 save 호출지점 갱신.
5. 각 패러다임 `resume` 경로에 `opt.restore(...)?` 추가.
6. 테스트
   - 옵티마이저별 단위 테스트: `step → snapshot → new_instance →
     restore → 다음 step 결과 동일` (bit-identical).
   - Phase 4.1 의 왕복 테스트 중 최소 1건(Adam 사용)을 **loss bit-identical**
     기준으로 조임.

### 변경 파일 요약

```
src/optimizer/mod.rs        (trait 확장, Snapshot/Kind 정의)
src/optimizer/sgd.rs        (snapshot = empty)
src/optimizer/momentum.rs   (velocity 직렬화)
src/optimizer/adagrad.rs    (sum_sq 직렬화)
src/optimizer/rmsprop.rs    (ema_sq 직렬화)
src/optimizer/adam.rs       (m, v, t 직렬화)
src/optimizer/adamw.rs      (m, v, t + wd 직렬화)
src/trainer/checkpoint.rs   (필드 + 헬퍼 시그니처 갱신)
src/trainer/supervised.rs, unsupervised.rs, autoregressive.rs,
  semi_supervised.rs       (save/resume 경로에 optimizer 연결)
```

### 검증

- 옵티마이저 단위 테스트 6건 추가 → 191 passed.
- Phase 4.1 의 왕복 테스트 중 Adam 사용 케이스를 `assert_approx_eq!(loss_full,
  loss_resumed, tolerance = 1e-6)` 로 조임.

### 리스크

- **`Vec<Vec<f32>>` JSON 직렬화 크기**: 큰 모델(파라미터 수 백만)에서 JSON
  이 수십 MB 될 수 있음. 체크포인트는 drain-state 이므로 1회성이라 허용.
  성능 문제 시 bincode 로 이전 고려 (후속 페이즈).
- **파라미터 순서 불일치**: `register` 순서가 세이브/로드 시점에 같아야 함.
  현재 구조상 사용자가 `register` 를 명시적으로 호출하므로, **호출 순서만
  같으면** 버퍼 인덱스가 일치. → 테스트에서 이 계약을 문서화.
- **`OptimizerKind` 불일치**: `restore` 시 kind 검사 실패 → `MlResult::Err`.
  사용자가 optimizer 타입을 바꾸고 resume 시도하는 실수를 방지.

### 범위 외

- Learning rate scheduler 상태 복원 (현재 scheduler 모듈 자체가 없음).
- 분산 학습에서의 파라미터 그룹별 LR 스냅샷.

---

## Phase 7 — RL 트레이너 Ctrl+C + 체크포인트

### 목적

RL 은 현재 `num_episodes` 루프 안에서 Ctrl+C 감지/체크포인트 저장이 없어
장시간 학습 중단 시 진행률이 날아간다. 다른 패러다임과 동등한 UX 제공.

### 현재 상태

- `RLTrainer::fit` 은 에피소드 루프를 직접 돌며 `ComputationGraph::reset_graph()`
  / 롤아웃 / 반환 계산 / 정책 그래디언트 / `optimizer.step()` 을 수행.
- `TrainerCore::run_epoch` 의 배치 루프 시맨틱과 맞지 않아 Phase 2 범위 외로 유지됨.
- `checkpoint::interrupt_flag()` 와 `save_interrupt_checkpoint` 헬퍼는 이미
  있으므로 RL 에서도 재사용 가능.

### 설계

배치 루프가 아니라 **에피소드 단위 감지**. 각 에피소드 끝에서 `flag.load()`
확인:

```rust
// 에피소드 루프 내부
let interrupt = if cfg.checkpoint_dir.is_some() {
    let f = interrupt_flag();
    clear_interrupt(&f);
    Some(f)
} else { None };

for episode in 0..num_episodes {
    // … rollout, grad, step …

    if let Some(ref flag) = interrupt {
        if flag.load(Ordering::Relaxed) && confirm_interrupt() {
            // 체크포인트 저장 (paradigm: Reinforcement)
            save_interrupt_checkpoint(...)?;
            return Ok(TrainResult { interrupted: true, … });
        }
    }
}
```

### 할 일

1. `RLTrainer::fit` 에 `cfg.checkpoint_dir` 가 있을 때 interrupt flag 설정.
2. 에피소드 루프 말미에 flag 검사 → 저장 → return.
3. `RLTrainer::resume(…, checkpoint_path)` 구현.
4. `ParadigmTag::Reinforcement` 확인 로직 추가.
5. 체크포인트 저장 시 **에피소드 진행률**(`episode_done`, `last_episode_ret`)을
   기록. 현재 `TrainingCheckpoint.epochs_done` 를 재활용 가능.
6. 테스트: Phase 4.1 의 왕복 테스트 패턴으로 RL 1건 추가.

### 변경 파일

```
src/trainer/reinforcement.rs   (Ctrl+C, save/resume 로직 추가)
src/tests/common/model/reinforcement/mod.rs  (왕복 테스트)
```

### 검증

- RL 왕복 테스트 1건 추가 → 192 passed (Phase 6 완료 후 기준).
- 기존 RL 테스트(`rl_learns_cartpole_stub` 등) 회귀 없음.

### 리스크

- **에피소드 중간 중단은 지원하지 않음**. 한 에피소드 끝에서만 체크 — 장기
  에피소드(예: 10k steps) 에서 반응이 느려 보일 수 있음. 기록상 제한으로 남김.
- Phase 6 이 선행되지 않으면 `optimizer_snapshot` 미저장으로 분산 감소
  baseline 버퍼가 없어 resume 손실이 꽤 클 수 있음 (REINFORCE + baseline).
  Phase 6 완료 후 Phase 7 을 실행 권장.

### 범위 외

- PPO / A2C 등 on-policy batch 알고리즘으로의 확장.
- 리플레이 버퍼 직렬화 (off-policy RL 도입 시점).

---

## Phase B1 — 기타 loss 의 per-row 버그 감사

### 목적

Phase B0 에서 `SoftmaxCrossEntropyLoss` 의 per-row 버그가 수정되었지만,
동일 패턴 버그가 `CrossEntropyLoss`, `BinaryCrossEntropyLoss`, `MSE`, `MAE`,
`Huber` 중 한 곳 이상에 잠복할 가능성 존재.

### 할 일

1. 각 손실 함수의 `forward/backward` 를 읽고, B>1 입력 `[B, V]` 또는
   `[B, D]` 를 **단일 분포**로 취급하는지 확인.
2. 의심 케이스 각각에 대해:
   - B=1 회귀 테스트 (기존 동작 유지 확인).
   - B>1 단위 테스트:
     - 출력이 수치적으로 올바른지(외부 reference 값과 비교).
     - 원소별 손실의 평균이 반환되는지.
     - `backward` 의 gradient 가 `p - t` 또는 `2(y-t)/BD` 같은 per-element
       공식을 따르는지.
3. 버그 발견 시 B0 와 동일한 per-row 수정.

### 의심 우선순위

우선순위 기준: "분포의 합/정규화가 필요한 연산" > "독립 원소 연산".

1. **CrossEntropyLoss (high)** — SoftmaxCE 와 같은 클래스 분포 구조.
2. **BinaryCrossEntropyLoss (high)** — per-element 이지만 sum-reduce 경로
   확인 필요.
3. **MSE / MAE (medium)** — per-element 이므로 단순 평균이면 OK, reduce
   축 오류만 점검.
4. **Huber (medium)** — piecewise 이지만 per-element. 동.

### 변경 파일

```
src/loss/function.rs                      (수정 대상)
src/tests/loss_per_row.rs (신규) 또는
src/loss/function.rs::tests (확장)
```

### 검증

- 감사 후 수정이 없다면 "검증된 손실" 주석만 추가.
- 수정 시 B=1/B>1 회귀 테스트 2건씩 추가.

### 리스크

- 수정이 gradient shape 을 바꾸면 기존 학습 루프에서 rank 오류 발생 가능.
  → 각 손실을 사용하는 모델(softmax, logistic, MLP 등)의 기존 테스트로 먼저
  회귀 감시.

### 범위 외

- GPU 백엔드 동등성 확인 (현재 CPU-only 테스트만 신뢰).

---

## 2. 실행 순서 권고

```
[즉시]         Phase 8 (문서 동기화, <1h)
[Wave 1]       Phase 3.1 (훅 일원화) → Phase 4.1 (왕복 테스트)
[Wave 2]       Phase 6 (Optimizer snapshot) → Phase 4.1 tolerance 조임
[Wave 3]       Phase 7 (RL 체크포인트)
[Wave 4]       Phase B1 (loss 감사)
```

의존성 그래프:

```
Phase 8  ──────────────┐
                       │
Phase 3.1 ─────────────┤
                       ├─→ Phase 4.1 ─→ Phase 6 ─→ Phase 7
                       │
Phase B1 ──────────────┘  (독립, 어느 시점에도 가능)
```

Phase 6 은 규모가 크므로 **한 커밋에 묶지 말고** `OptimizerSnapshot` 정의 →
stateless/simple 옵티마이저 구현 → Adam/AdamW → 트레이너 통합 순으로
다단계 커밋 권장.

---

## 3. 공통 검증 커맨드

```bash
# 전체 테스트
cargo test --lib --features enableBackward

# 페이즈별 스코프
cargo test --lib --features enableBackward trainer::       # 트레이너 계열
cargo test --lib --features enableBackward optimizer::     # Phase 6
cargo test --lib --features enableBackward loss::          # Phase B1
cargo test --lib --features enableBackward checkpoint      # Phase 4.1

# 컴파일 위생
cargo check --lib --features enableBackward
cargo clippy --lib --features enableBackward -- -D warnings  # 선택적
```

---

## 4. 각 페이즈 종료 시 체크리스트

- [ ] `cargo test --lib --features enableBackward` **grandtotal 증가 확인**
      (회귀 없음 + 신규 테스트 반영).
- [ ] 신규 경고 0건 (`cargo build` 출력).
- [ ] 이 문서의 해당 페이즈 섹션에 "완료" 표기.
- [ ] `TRAINER_PROGRESS_REPORT.md` §1 에 요약 절 추가.
- [ ] 사용자 불변 지침 준수 재확인:
      `LatentDiffusion/Encoder/Decoder/Scheduler` 스텁 보존.

---

## 5. 불변 지침 (원 플랜 유지)

- **LatentDiffusion / Encoder / Decoder / Scheduler 래퍼는 제거 금지.**
  `#[allow(dead_code)]` + TODO 주석으로 보존.
- 페이즈는 순서대로 실행하고, 각 페이즈 종료 시 평가 후 다음으로.
- 작업 중 발견한 버그는 별도 `B-` 페이즈로 기록.
