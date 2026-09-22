/// 학습 루프에서 활성화할 메트릭 집합.
///
/// 모든 메서드가 `const fn`이므로 `const` 변수로 컴파일 타임에 확정 가능.
///
/// # 예시
/// ```no_run
/// use trench_deep::trainer::Metrics;
///
/// let m = Metrics::none().grad_norm().accuracy();
/// ```
#[derive(Clone, Copy)]
pub struct Metrics {
    /// 현재 패러다임의 대표 메트릭(accuracy, perplexity, return, lambda 등)
    pub paradigm: bool,
    /// 전체 파라미터의 그래디언트 L2 노름
    pub grad_norm: bool,
    /// Update Ratio = ||lr·grad|| / ||W||  (학습률 스케일 진단용)
    pub update_ratio: bool,
    /// argmax 기반 분류 정확도
    pub accuracy: bool,
    /// Forward / Backward 패스 소요 시간
    pub fw_bw_timing: bool,
}

impl Metrics {
    /// 모든 메트릭 비활성.
    pub const fn none() -> Self {
        Self {
            paradigm: false,
            grad_norm: false,
            update_ratio: false,
            accuracy: false,
            fw_bw_timing: false,
        }
    }

    /// 모든 메트릭 활성.
    pub const fn all() -> Self {
        Self {
            paradigm: true,
            grad_norm: true,
            update_ratio: true,
            accuracy: true,
            fw_bw_timing: true,
        }
    }

    pub const fn paradigm(mut self) -> Self {
        self.paradigm = true;
        self
    }
    pub const fn grad_norm(mut self) -> Self {
        self.grad_norm = true;
        self
    }
    pub const fn update_ratio(mut self) -> Self {
        self.update_ratio = true;
        self
    }
    pub const fn accuracy(mut self) -> Self {
        self.accuracy = true;
        self
    }
    pub const fn fw_bw_timing(mut self) -> Self {
        self.fw_bw_timing = true;
        self
    }

    pub const fn without_paradigm(mut self) -> Self {
        self.paradigm = false;
        self
    }
    pub const fn without_grad_norm(mut self) -> Self {
        self.grad_norm = false;
        self
    }
    pub const fn without_update_ratio(mut self) -> Self {
        self.update_ratio = false;
        self
    }
    pub const fn without_accuracy(mut self) -> Self {
        self.accuracy = false;
        self
    }
    pub const fn without_fw_bw_timing(mut self) -> Self {
        self.fw_bw_timing = false;
        self
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::none().paradigm()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 내부 최종 설정 구조체 (Trainer configuration 변환 결과)
// ────────────────────────────────────────────────────────────────────────────

/// Trainer 내부에서 사용하는 확정된 로그·메트릭 설정.
/// Created by the typed Trainer constructors.
pub struct LogConfig {
    /// 몇 배치마다 메트릭 계산 및 배치 progress bar 를 갱신할지 여부.
    /// `usize::MAX` = 배치 레벨 로그 완전 비활성.
    pub batch_log_interval: usize,

    /// 학습 완료 후 발행할 배치 요약의 간격.
    /// `usize::MAX`면 완료 후 배치 로그를 남기지 않는다.
    pub batch_summary_interval: usize,

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

    /// Base seed for deterministic shuffling and sampling.
    pub seed: u64,
}

pub type TrainerConfig = LogConfig;

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            batch_log_interval: 1,
            batch_summary_interval: usize::MAX,
            epoch_log_interval: 10,
            nan_check_interval: 1,
            metrics: Metrics::default(),
            show_progress: true,
            checkpoint_dir: None,
            seed: 0,
        }
    }
}
