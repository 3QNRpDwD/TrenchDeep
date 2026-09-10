use trench_deep::{nn::*, trainer::TrainableModel, ExecutionContext, MlResult, Variable};
use trench_deep::optimizer::{Adam, Optimizer};
use trench_deep::trainer::{
    ConsistencyRamp, EpochSchedule, SemiSupervisedDataset, SemiSupervisedTrainer,
};

#[test]
fn supplied_views_reject_broadcasting_and_empty_batches_before_recording() -> MlResult<()> {
    let context = ExecutionContext::new();
    let model = PiClassifier::new(&context, 2, 2, 0.1)?;
    let input = context.input(vec![1.0, 2.0], &[1, 2])?;
    let target = context.tensor(vec![1.0, 0.0], &[1, 2])?;
    let other = context.input(vec![1.0; 4], &[2, 2])?;
    let empty = context.input(vec![], &[0, 2])?;
    for (first, second) in [(&input, &other), (&empty, &empty)] {
        assert!(matches!(
            model.forward_loss_with_augmentations(&input, &target, first, second, 0.4),
            Err(trench_deep::MlError::TensorError(
                trench_deep::TensorError::InvalidOperation { op: "pi_model", .. }
            ))
        ));
        assert_eq!(context.graph_stats()?.graph_nodes, 0);
    }
    Ok(())
}

#[test]
fn pi_model_pilot_trains_end_to_end() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = PiClassifier::new(&context, 2, 2, 0.1)?;
    let labeled = [
        context.input(vec![1.0, 1.0], &[1, 2])?,
        context.input(vec![-1.0, -1.0], &[1, 2])?,
    ];
    let targets = [
        context.tensor(vec![1.0, 0.0], &[1, 2])?,
        context.tensor(vec![0.0, 1.0], &[1, 2])?,
    ];
    let unlabeled = [
        context.input(vec![0.9, 1.1], &[1, 2])?,
        context.input(vec![-0.9, -1.1], &[1, 2])?,
    ];
    let labeled_refs = labeled.iter().collect::<Vec<_>>();
    let target_refs = targets.iter().collect::<Vec<_>>();
    let unlabeled_refs = unlabeled.iter().collect::<Vec<_>>();
    let dataset =
        SemiSupervisedDataset::new(&context, &labeled_refs, &target_refs, &unlabeled_refs)?;
    let mut optimizer = Adam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let result = SemiSupervisedTrainer::silent(&context)
        .with_ramp(ConsistencyRamp::Sigmoid {
            max_weight: 1.0,
            ramp_epochs: 2,
        })
        .fit(
            &mut model,
            &mut optimizer,
            &dataset,
            EpochSchedule::new(3)?.with_tolerance(0.0),
        )?;
    assert!(result.final_loss.is_finite());
    assert_eq!(context.graph_stats()?.graph_nodes, 0);
    Ok(())
}
