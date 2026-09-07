#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use trench_deep::{
    nn::{LinearPolicy, TwoArmedBandit},
    optimizer::{Optimizer, SGD},
    trainer::*,
    *,
};
struct StopAfterStep {
    ctx: ExecutionContext,
}
impl TrainingObserver for StopAfterStep {
    fn on_batch_end(&mut self, _: &BatchEndContext) {
        assert_eq!(self.ctx.graph_stats().expect("context").graph_nodes, 0);
        checkpoint::request_interrupt();
    }
}
#[test]
fn rl_interrupt_saves_completed_episode_after_scope_cleanup()
-> Result<(), Box<dyn std::error::Error>> {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            checkpoint::clear_interrupt();
        }
    }
    let _clear = Clear;
    checkpoint::clear_interrupt();
    let directory = std::env::temp_dir().join(format!(
        "trench-rl-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let ctx = ExecutionContext::new();
    let mut model = LinearPolicy::new(&ctx, 1, 2)?;
    let mut environment = TwoArmedBandit::default().with_seed(91);
    let mut optimizer = SGD::new(&ctx, 0.01)?;
    optimizer.register_all(&model.parameters())?;
    let trainer = RLTrainer::from_trainer(
        &ctx,
        Trainer::builder()
            .show_progress(false)
            .checkpoint_dir(directory.to_str().ok_or("path")?)
            .build(),
    )
    .with_observer(Box::new(StopAfterStep { ctx: ctx.clone() }));
    let result = trainer.fit_checkpointed(
        &mut model,
        &mut environment,
        &mut optimizer,
        EpisodeSchedule::new(5, 1)?,
    )?;
    assert_eq!(result.stop_reason, StopReason::Interrupted);
    assert_eq!(result.units_completed, 1);
    let paths = result.checkpoint.ok_or("missing checkpoint")?;
    let metadata = checkpoint::TrainingCheckpoint::load(&paths.metadata)?;
    assert_eq!(metadata.epochs_done, 1);
    assert_eq!(metadata.total_epochs, 5);
    assert_eq!(
        metadata.paradigm,
        Some(checkpoint::ParadigmTag::Reinforcement)
    );
    let mut restored = LinearPolicy::new(&ctx, 1, 2)?;
    restored.load_checkpoint(&paths.model)?;
    for (p, q) in model.parameters().iter().zip(restored.parameters()) {
        assert_eq!(p.tensor().to_vec()?, q.tensor().to_vec()?);
        assert!(p.grad()?.is_none());
    }
    std::fs::remove_file(paths.model)?;
    std::fs::remove_file(paths.metadata)?;
    std::fs::remove_dir(directory)?;
    Ok(())
}
