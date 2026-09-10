use super::*;

#[test]
fn default_preset_auto_attaches_accuracy_hook() {
    let trainer = SupervisedTrainer::default();
    assert_eq!(
        trainer.core.hook_count(),
        1,
        "default() 는 ClassificationAccuracy 훅을 자동 장착해야 함"
    );
}

#[test]
fn verbose_preset_auto_attaches_accuracy_hook() {
    let trainer = SupervisedTrainer::verbose();
    assert_eq!(
        trainer.core.hook_count(),
        1,
        "verbose() 는 ClassificationAccuracy 훅을 자동 장착해야 함"
    );
}

#[test]
fn minimal_preset_has_no_paradigm_hook() {
    let trainer = SupervisedTrainer::minimal();
    assert_eq!(
        trainer.core.hook_count(),
        0,
        "minimal() 은 핵심 메트릭만 표시하므로 accuracy 훅이 없어야 함"
    );
}

#[test]
fn silent_preset_has_no_auto_hooks() {
    let trainer = SupervisedTrainer::silent();
    assert_eq!(
        trainer.core.hook_count(),
        0,
        "silent() 은 metrics.accuracy=false → 훅 없음"
    );
}

#[test]
fn builder_without_accuracy_metric_skips_auto_hook() {
    let trainer: SupervisedTrainer = Trainer::builder()
        .metrics(Metrics::none().grad_norm())
        .build()
        .into();
    assert_eq!(
        trainer.core.hook_count(),
        0,
        "metrics.accuracy=false 빌더 경로는 자동 훅이 없어야 함"
    );
}

#[test]
fn builder_with_accuracy_metric_auto_attaches_hook() {
    let trainer: SupervisedTrainer = Trainer::builder()
        .metrics(Metrics::none().accuracy())
        .build()
        .into();
    assert_eq!(
        trainer.core.hook_count(),
        1,
        "metrics.accuracy=true 빌더 경로는 자동 훅이 장착되어야 함"
    );
}

#[test]
fn from_config_raw_path_has_no_auto_hooks() {
    // metrics.accuracy=true 여도 from_config 경로는 자동 훅을 달지 않는다 (계약).
    let cfg = LogConfig {
        batch_log_interval: usize::MAX,
        batch_summary_interval: usize::MAX,
        epoch_log_interval: usize::MAX,
        nan_check_interval: usize::MAX,
        metrics: Metrics::all(),
        show_progress: false,
        checkpoint_dir: None,
        seed: 0,
    };
    let trainer = SupervisedTrainer::from_config(cfg);
    assert_eq!(
        trainer.core.hook_count(),
        0,
        "from_config 원시 경로는 훅 자동 장착 대상이 아니다"
    );
}
