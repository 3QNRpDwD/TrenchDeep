use super::*;

/// 학습 루프에서 활성화할 메트릭 집합.
///
/// 모든 메서드가 `const fn`이므로 `const` 변수로 컴파일 타임에 확정 가능.
///
/// # 예시
/// ```no_run
/// let m = Metrics::none().grad_norm().accuracy();
/// ```
#[derive(Clone, Copy)]
pub struct Metrics {
    /// 전체 파라미터의 그래디언트 L2 노름
    pub grad_norm:    bool,
    /// Update Ratio = ||lr·grad|| / ||W||  (학습률 스케일 진단용)
    pub update_ratio: bool,
    /// argmax 기반 분류 정확도
    pub accuracy:     bool,
    /// Forward / Backward 패스 소요 시간
    pub fw_bw_timing: bool,
}

impl Metrics {
    /// 모든 메트릭 비활성.
    pub const fn none() -> Self {
        Self { grad_norm: false, update_ratio: false, accuracy: false, fw_bw_timing: false }
    }

    /// 모든 메트릭 활성.
    pub const fn all() -> Self {
        Self { grad_norm: true, update_ratio: true, accuracy: true, fw_bw_timing: true }
    }

    pub const fn grad_norm(mut self)    -> Self { self.grad_norm    = true; self }
    pub const fn update_ratio(mut self) -> Self { self.update_ratio = true; self }
    pub const fn accuracy(mut self)     -> Self { self.accuracy     = true; self }
    pub const fn fw_bw_timing(mut self) -> Self { self.fw_bw_timing = true; self }

    pub const fn without_grad_norm(mut self)    -> Self { self.grad_norm    = false; self }
    pub const fn without_update_ratio(mut self) -> Self { self.update_ratio = false; self }
    pub const fn without_accuracy(mut self)     -> Self { self.accuracy     = false; self }
    pub const fn without_fw_bw_timing(mut self) -> Self { self.fw_bw_timing = false; self }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::none().grad_norm().accuracy().fw_bw_timing()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 내부 최종 설정 구조체 (TrainerBuilder → Trainer 변환 결과)
// ────────────────────────────────────────────────────────────────────────────

/// Trainer 내부에서 사용하는 확정된 로그·메트릭 설정.
/// `TrainerBuilder::build()`가 생성.
pub struct LogConfig {
    /// 몇 배치마다 메트릭 계산 및 배치 progress bar 를 갱신할지 여부.
    /// `usize::MAX` = 배치 레벨 로그 완전 비활성.
    pub batch_log_interval: usize,

    /// 몇 에폭마다 에폭 레벨 로그를 출력할지.
    pub epoch_log_interval: usize,

    /// 몇 배치마다 NaN/Inf 검사를 수행할지.
    /// `usize::MAX` = NaN 검사 비활성 (성능 우선, 비권장).
    pub nan_check_interval: usize,

    /// 활성화된 메트릭 집합.
    pub metrics: Metrics,

    /// progress bar 출력 여부.
    pub show_progress: bool,

    /// 체크포인트 저장 디렉토리.
    /// `None`이면 인터럽트 시 체크포인트를 저장하지 않는다.
    pub checkpoint_dir: Option<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// Builder
// ────────────────────────────────────────────────────────────────────────────

/// `Trainer`를 구성하는 빌더.
///
/// # 예시
/// ```no_run
/// let trainer = Trainer::builder()
///     .log_every_n_batches(50)
///     .nan_check(true)
///     .metrics(Metrics::none().grad_norm().accuracy())
///     .show_progress(true)
///     .build();
/// ```
pub struct TrainerBuilder {
    batch_log_interval: usize,
    epoch_log_interval: usize,
    nan_check_interval: usize,
    metrics:            Metrics,
    show_progress:      bool,
    checkpoint_dir:     Option<String>,
}

impl TrainerBuilder {
    pub fn new() -> Self {
        Self {
            batch_log_interval: 1,
            epoch_log_interval: 1,
            nan_check_interval: 1,
            metrics:            Metrics::default(),
            show_progress:      true,
            checkpoint_dir:     None,
        }
    }

    /// 몇 배치마다 메트릭을 계산하고 로그를 출력할지 설정.
    ///
    /// `0`을 입력하면 배치 레벨 로그가 완전히 비활성화.
    /// 예: `50` → 50배치마다 grad_norm 계산 + progress bar 갱신.
    pub fn log_every_n_batches(mut self, n: usize) -> Self {
        self.batch_log_interval = if n == 0 { usize::MAX } else { n };
        self
    }

    /// 몇 에폭마다 에폭 요약 로그를 출력할지 설정.
    pub fn log_every_n_epochs(mut self, n: usize) -> Self {
        self.epoch_log_interval = if n == 0 { usize::MAX } else { n };
        self
    }

    /// NaN/Inf 그래디언트 검사 활성화 여부.
    ///
    /// `false`로 설정하면 성능이 향상되지만 발산 감지가 불가능.
    /// 완전히 검증된 모델·학습률 조합에서만 비활성화를 권장.
    pub fn nan_check(mut self, enabled: bool) -> Self {
        self.nan_check_interval = if enabled { 1 } else { usize::MAX };
        self
    }

    /// 활성화할 메트릭 집합을 설정.
    ///
    /// ```no_run
    /// .metrics(Metrics::none().grad_norm().accuracy())
    /// ```
    pub fn metrics(mut self, m: Metrics) -> Self {
        self.metrics = m;
        self
    }

    /// 터미널 progress bar 출력 여부.
    pub fn show_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// 체크포인트 저장 디렉토리를 설정한다.
    ///
    /// 설정하면 학습 중 Ctrl+C 인터럽트 시 모델 가중치와 학습 상태를
    /// 이 디렉토리에 저장한다. `resume()`으로 중단 지점부터 재개 가능.
    ///
    /// ```no_run
    /// .checkpoint_dir("checkpoints/my_model")
    /// ```
    pub fn checkpoint_dir(mut self, dir: &str) -> Self {
        self.checkpoint_dir = Some(dir.to_string());
        self
    }

    /// 설정을 확정하고 `Trainer`를 생성.
    pub fn build(self) -> Trainer {
        Trainer {
            config: LogConfig {
                batch_log_interval: self.batch_log_interval,
                epoch_log_interval: self.epoch_log_interval,
                nan_check_interval: self.nan_check_interval,
                metrics:            self.metrics,
                show_progress:      self.show_progress,
                checkpoint_dir:     self.checkpoint_dir,
            },
        }
    }
}

impl Default for TrainerBuilder {
    fn default() -> Self { Self::new() }
}
