use trench_deep::{nn::*, trainer::TrainableModel, ExecutionContext, MlResult, Variable};
use trench_deep::optimizer::{Adam, Optimizer};
use trench_deep::trainer::{EpisodeSchedule, RLTrainer};

#[test]
fn bandit_policy_pilot_trains_end_to_end() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = LinearPolicy::new(&context, 1, 2)?;
    let mut environment = TwoArmedBandit::default();
    environment.mean_rewards = [-1.0, 1.0];
    environment.noise_scale = 0.0;
    let mut optimizer = Adam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let result = RLTrainer::silent(&context)
        .with_seed(19)
        .with_baseline(false)
        .fit(
            &mut model,
            &mut environment,
            &mut optimizer,
            EpisodeSchedule::new(20, 1)?,
        )?;
    assert!(result.final_loss.is_finite());
    assert_eq!(result.units_completed, 20);
    assert_eq!(context.graph_stats()?.graph_nodes, 0);
    Ok(())
}
