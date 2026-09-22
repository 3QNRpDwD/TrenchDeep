use super::*;

struct Hook;
impl MetricHook for Hook {
    fn update(&mut self, _: &BatchContext<'_>) -> MlResult<()> {
        Ok(())
    }
    fn compute(&self) -> f32 {
        0.0
    }
    fn reset(&mut self) -> MlResult<()> {
        Ok(())
    }
    fn name(&self) -> &str {
        "retained"
    }
}
struct Observer;
impl TrainingObserver for Observer {}

fn settings<M, S>(trainer: &Trainer<M, S>) -> (usize, usize, usize, usize, [bool; 5], bool) {
    let c = &trainer.service.core.config;
    let m = c.metrics;
    (
        c.batch_log_interval,
        c.batch_summary_interval,
        c.epoch_log_interval,
        c.nan_check_interval,
        [
            m.paradigm,
            m.grad_norm,
            m.update_ratio,
            m.accuracy,
            m.fw_bw_timing,
        ],
        c.show_progress,
    )
}

fn check_presets<M, S>(trainer: Trainer<M, S>) {
    let off = usize::MAX;
    let trainer = trainer.verbose();
    assert_eq!(settings(&trainer), (1, 100, 1, 1, [true; 5], true));
    let trainer = trainer.silent();
    assert_eq!(settings(&trainer), (off, off, off, off, [false; 5], false));
    let trainer = trainer.minimal();
    assert_eq!(settings(&trainer), (1, off, 10, 1, [false; 5], true));
    let trainer = trainer.default();
    assert_eq!(
        settings(&trainer),
        (1, off, 10, 1, [true, false, false, false, false], true)
    );
    let trainer = trainer
        .metrics(Metrics::none().accuracy())
        .log_every_n_epochs(3)
        .nan_check(false);
    assert_eq!(
        settings(&trainer),
        (1, off, 3, off, [false, false, false, true, false], true)
    );
    let trainer = trainer.minimal();
    assert_eq!(settings(&trainer), (1, off, 10, 1, [false; 5], true));
}

#[test]
fn every_strategy_and_mode_supports_all_presets_and_last_setting_wins() {
    let ctx = ExecutionContext::new();
    check_presets(Trainer::supervised(&ctx));
    check_presets(Trainer::autoregressive(&ctx));
    check_presets(Trainer::diffusion(&ctx));
    check_presets(Trainer::semi_supervised(&ctx));
    check_presets(Trainer::supervised(&ctx).prepared());
    check_presets(Trainer::autoregressive(&ctx).prepared());
    check_presets(Trainer::diffusion(&ctx).prepared());
    check_presets(Trainer::semi_supervised(&ctx).prepared());
}

#[test]
fn preset_and_mode_changes_retain_configuration_and_rng_position() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let trainer = Trainer::semi_supervised(&ctx)
        .with_seed(81)
        .checkpoint_dir("retained-checkpoint")
        .with_max_grad_norm(0.5)?
        .with_noise_scale(0.25)?
        .with_ramp(ConsistencyRamp::Constant(0.4))
        .with_hook(Box::new(Hook))
        .with_observer(Box::new(Observer));
    let reference = TrainingRuntime::new(81);
    let mut a: Vec<_> = (0..32).collect();
    let mut b = a.clone();
    trainer.service.core.runtime.shuffle(&mut a);
    reference.shuffle(&mut b);
    assert_eq!(a, b);
    let trainer = trainer.silent().minimal().prepared().verbose().default();
    trainer.service.core.runtime.shuffle(&mut a);
    reference.shuffle(&mut b);
    assert_eq!(a, b);
    assert_eq!(trainer.service.context.id(), ctx.id());
    assert_eq!(trainer.service.core.config.seed, 81);
    assert_eq!(
        trainer.service.core.config.checkpoint_dir.as_deref(),
        Some("retained-checkpoint")
    );
    assert_eq!(trainer.service.max_grad_norm, Some(0.5));
    assert_eq!(trainer.strategy.noise_scale, 0.25);
    assert_eq!(trainer.ramp.value(7), 0.4);
    assert_eq!(trainer.service.core.hook_count(), 1);
    assert_eq!(trainer.service.core.observer_count(), 1);
    assert!(
        Trainer::semi_supervised(&ctx)
            .with_noise_scale(f32::NAN)
            .is_err()
    );
    assert!(
        Trainer::semi_supervised(&ctx)
            .prepared()
            .with_noise_scale(-1.0)
            .is_err()
    );
    Ok(())
}
