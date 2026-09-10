use super::*;

#[test]
fn logging_presets_keep_metric_layers_separate() {
    let silent = Trainer::silent().core;
    assert!(!silent.config().show_progress);
    assert_eq!(silent.config().batch_log_interval, usize::MAX);
    assert_eq!(silent.config().epoch_log_interval, usize::MAX);
    assert!(!silent.config().metrics.paradigm);

    let minimal = Trainer::minimal().core;
    assert!(minimal.config().show_progress);
    assert_eq!(minimal.config().batch_log_interval, 1);
    assert_eq!(minimal.config().batch_summary_interval, usize::MAX);
    assert_eq!(minimal.config().epoch_log_interval, 10);
    assert!(!minimal.config().metrics.paradigm);
    assert!(!minimal.config().metrics.grad_norm);
    assert!(!minimal.config().metrics.update_ratio);
    assert!(!minimal.config().metrics.fw_bw_timing);

    let default = Trainer::default().core;
    assert!(default.config().metrics.paradigm);
    assert!(!default.config().metrics.grad_norm);
    assert!(!default.config().metrics.update_ratio);
    assert!(!default.config().metrics.fw_bw_timing);
    assert_eq!(default.config().batch_summary_interval, usize::MAX);
    assert_eq!(default.config().epoch_log_interval, 10);

    let verbose = Trainer::verbose().core;
    assert!(verbose.config().metrics.paradigm);
    assert!(verbose.config().metrics.grad_norm);
    assert!(verbose.config().metrics.update_ratio);
    assert!(verbose.config().metrics.accuracy);
    assert!(verbose.config().metrics.fw_bw_timing);
    assert_eq!(verbose.config().batch_summary_interval, 100);
    assert_eq!(verbose.config().epoch_log_interval, 1);
}
