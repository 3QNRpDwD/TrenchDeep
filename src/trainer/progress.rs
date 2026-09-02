#[allow(unused_imports)]
use super::*;

// progress 전용 import
use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

// ────────────────────────────────────────────────────────────────────────────
// EpochProgress — 에폭 레벨 progress bar 래퍼
// ────────────────────────────────────────────────────────────────────────────

/// 에폭 레벨 progress bar 를 관리.
///
/// `show = false`일 때 hidden ProgressBar를 반환하므로 호출부에 분기 없음.
pub(crate) struct EpochProgress {
    multi:     MultiProgress,
    epoch_bar: ProgressBar,
    show:      bool,
}

impl EpochProgress {
    pub fn new(epochs: usize, show: bool) -> Self {
        // 연산 trace와 동적 progress 출력은 동일 터미널 행을 공유할 수 없다.
        // debugging 빌드에서는 구조화 로그를 우선하고 progress만 숨긴다.
        let show = show && !cfg!(feature = "debugging");
        // 짧은 배치도 에폭 바와 동시에 보이도록 기본 15Hz보다 자주 그린다.
        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(60));
        let epoch_bar = if show {
            let pb = multi.add(ProgressBar::new(epochs as u64));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] \
                         [ {wide_bar:.cyan/blue} ] {percent:>3}% Epochs ({eta}) | {msg}"
                    )
                    .unwrap()
                    .progress_chars("▉ "),
            );
            pb
        } else {
            ProgressBar::hidden()
        };
        Self { multi, epoch_bar, show }
    }

    /// 배치 레벨 progress bar 를 에폭 바 아래에 추가하여 반환.
    pub fn start_batch_bar(
        &self,
        epoch:     usize,
        epochs:    usize,
        n_batches: usize,
    ) -> BatchProgress {
        if !self.show {
            return BatchProgress { bar: ProgressBar::hidden(), active: false };
        }
        // MultiProgress는 각 bar가 한 번 draw되어야 해당 행을 합성한다.
        // 첫 에폭부터 에폭/배치 두 행이 함께 보이도록 에폭 상태를 먼저 등록한다.
        self.epoch_bar.tick();
        let epoch_percent = (epoch + 1) * 100 / epochs.max(1);
        let template = format!(
            "  > Epoch {:>3}% \
             [ {{wide_bar:.green/blue}} ] {{percent:>3}}% Batches ({{eta}}) | {{msg}}",
            epoch_percent,
        );
        let bar = self.multi.add(ProgressBar::new(n_batches as u64));
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&template)
                .unwrap()
                .progress_chars("█ "),
        );
        // 배치 연산 중에는 상태 변경이 없으므로 ticker가 없으면 짧은 배치 바가
        // 첫 redraw 전에 finish_and_clear 되어 화면에 나타나지 않을 수 있다.
        bar.enable_steady_tick(Duration::from_millis(16));
        BatchProgress { bar, active: true }
    }

    pub fn set_msg(&self, msg: &str) {
        self.epoch_bar.set_message(msg.to_string());
    }

    pub fn inc(&self) {
        self.epoch_bar.inc(1);
    }

    pub fn finish_converged(&self) {
        self.epoch_bar.finish_with_message("Converged");
    }

    pub fn finish_completed(&self) {
        self.epoch_bar.finish_with_message("Completed");
    }

    pub fn abandon(&self, msg: &str) {
        self.epoch_bar.abandon_with_message(msg.to_string());
    }

    /// progress bar 출력을 일시 중지하고 클로저를 실행한다.
    ///
    /// 인터럽트 시 사용자 입력 프롬프트를 표시할 때 사용.
    /// 클로저 실행이 끝나면 progress bar가 자동으로 재개된다.
    pub fn suspend<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.multi.suspend(f)
    }

    pub fn finish_interrupted(&self) {
        self.epoch_bar.finish_with_message("Interrupted — checkpoint saved");
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BatchProgress — 배치 레벨 progress bar 래퍼
// ────────────────────────────────────────────────────────────────────────────

/// 배치 레벨 progress bar 를 관리.
///
/// `active = false`(hidden bar)이면 모든 메서드가 no-op 이므로
/// 호출부에 `if show_progress` 분기가 필요 없음.
pub(crate) struct BatchProgress {
    bar:    ProgressBar,
    active: bool,
}

impl BatchProgress {
    pub fn set_msg(&self, msg: &str) {
        if self.active {
            self.bar.set_message(msg.to_string());
        }
    }

    pub fn inc(&self) {
        self.bar.inc(1);
    }

    pub fn finish(&self) {
        if self.active {
            self.bar.disable_steady_tick();
            self.bar.finish_and_clear();
        }
    }

    pub fn abandon(&self, msg: &str) {
        if self.active {
            self.bar.disable_steady_tick();
            self.bar.abandon_with_message(msg.to_string());
        }
    }
}
