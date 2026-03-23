use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// ────────────────────────────────────────────────────────────────────────────
// EpochProgress — 에폭 레벨 progress bar 래퍼
// ────────────────────────────────────────────────────────────────────────────

/// 에폭 레벨 progress bar 를 관리한다.
///
/// `show = false`일 때 hidden ProgressBar를 반환하므로 호출부에 분기가 없다.
pub(crate) struct EpochProgress {
    multi:     MultiProgress,
    epoch_bar: ProgressBar,
    show:      bool,
}

impl EpochProgress {
    pub fn new(epochs: usize, show: bool) -> Self {
        let multi = MultiProgress::new();
        let epoch_bar = if show {
            let pb = multi.add(ProgressBar::new(epochs as u64));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] \
                         [ {wide_bar:.cyan/blue} ] {pos}/{len} Epochs ({eta}) | {msg}"
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

    /// 배치 레벨 progress bar 를 에폭 바 아래에 추가하여 반환한다.
    pub fn start_batch_bar(
        &self,
        epoch:     usize,
        epochs:    usize,
        n_batches: usize,
    ) -> BatchProgress {
        if !self.show {
            return BatchProgress { bar: ProgressBar::hidden(), active: false };
        }
        let template = format!(
            "  > Epoch {:>3}/{:<3} \
             [ {{wide_bar:.green/blue}} ] {{pos}}/{{len}} Batches ({{eta}}) | {{msg}}",
            epoch + 1,
            epochs
        );
        let bar = self.multi.add(ProgressBar::new(n_batches as u64));
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&template)
                .unwrap()
                .progress_chars("█ "),
        );
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
}

// ────────────────────────────────────────────────────────────────────────────
// BatchProgress — 배치 레벨 progress bar 래퍼
// ────────────────────────────────────────────────────────────────────────────

/// 배치 레벨 progress bar 를 관리한다.
///
/// `active = false`(hidden bar)이면 모든 메서드가 no-op 이므로
/// 호출부에 `if show_progress` 분기가 필요 없다.
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
            self.bar.finish_and_clear();
        }
    }

    pub fn abandon(&self, msg: &str) {
        if self.active {
            self.bar.abandon_with_message(msg.to_string());
        }
    }
}
