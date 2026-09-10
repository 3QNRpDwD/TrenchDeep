use trench_deep::{nn::*, trainer::TrainableModel, ExecutionContext, MlResult, Variable};
use trench_deep::optimizer::{Adam, Optimizer};
use trench_deep::trainer::{EpochSchedule, SupervisedDataset, SupervisedTrainer};

#[test]
fn mlp_pilot_trains_end_to_end_and_predicts_probabilities() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = Mlp::new(&context, 2, 4, 2)?;
    let inputs = [
        context.input(vec![0.0, 0.0], &[1, 2])?,
        context.input(vec![1.0, 1.0], &[1, 2])?,
    ];
    let targets = [
        context.tensor(vec![1.0, 0.0], &[1, 2])?,
        context.tensor(vec![0.0, 1.0], &[1, 2])?,
    ];
    let input_refs: Vec<_> = inputs.iter().collect();
    let target_refs: Vec<_> = targets.iter().collect();
    let dataset = SupervisedDataset::new(&context, &input_refs, &target_refs)?;
    let mut optimizer = Adam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;

    let result = SupervisedTrainer::new(&context).fit(
        &mut model,
        &mut optimizer,
        &dataset,
        EpochSchedule::new(20)?,
    )?;
    assert!(result.final_loss.is_finite());
    assert_eq!(result.units_completed, 20);
    assert_eq!(context.graph_stats()?.graph_nodes, 0);

    let probabilities = model.predict(inputs[0].tensor())?.to_vec()?;
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    Ok(())
}
