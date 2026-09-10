use super::*;

#[test]
fn default_preset_auto_attaches_perplexity_hook() {
    let trainer = AutoregressiveTrainer::default();
    assert_eq!(
        trainer.core.hook_count(),
        1,
        "default() 는 Perplexity 훅을 자동 장착해야 함"
    );
}

#[test]
fn verbose_preset_auto_attaches_perplexity_hook() {
    let trainer = AutoregressiveTrainer::verbose();
    assert_eq!(
        trainer.core.hook_count(),
        1,
        "verbose() 는 Perplexity 훅을 자동 장착해야 함"
    );
}

#[test]
fn minimal_preset_has_no_perplexity_hook() {
    let trainer = AutoregressiveTrainer::minimal();
    assert_eq!(
        trainer.core.hook_count(),
        0,
        "minimal() 은 핵심 메트릭만 표시하므로 PPL 훅이 없어야 함"
    );
}

#[test]
fn silent_preset_has_no_auto_hooks() {
    let trainer = AutoregressiveTrainer::silent();
    assert_eq!(
        trainer.core.hook_count(),
        0,
        "silent() 은 epoch_log_interval=MAX → 훅 없음 (최대 성능 모드)"
    );
}

#[test]
fn from_config_raw_path_has_no_auto_hooks() {
    // epoch_log_interval 이 유효해도 from_config 경로는 자동 장착 안 함.
    let cfg = LogConfig {
        batch_log_interval: 1,
        batch_summary_interval: 100,
        epoch_log_interval: 1,
        nan_check_interval: 1,
        metrics: Metrics::default(),
        show_progress: false,
        checkpoint_dir: None,
        seed: 0,
    };
    let trainer = AutoregressiveTrainer::from_config(cfg);
    assert_eq!(trainer.core.hook_count(), 0);
}

#[test]
fn with_perplexity_shortcut_adds_single_hook_from_silent() {
    let trainer = AutoregressiveTrainer::silent().with_perplexity();
    assert_eq!(
        trainer.core.hook_count(),
        1,
        "silent() + with_perplexity() 는 훅 1개가 되어야 함"
    );
}
