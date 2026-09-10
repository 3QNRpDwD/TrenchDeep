use super::*;
use crate::legacy::{
    optimizer::{Optimizer, SGD},
    trainer::{
        DataLoader, DatasetBuilder, EpochSchedule, MemorySource, SupervisedSample,
        SupervisedStackCollator, Trainer,
    },
};

#[test]
fn supervised_trainer_accepts_built_loader() -> MlResult<()> {
    let dataset = DatasetBuilder::from_source(MemorySource::new(vec![
        ([0.0_f32, 0.0], [0.0_f32]),
        ([1.0, 0.0], [1.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 1.0], [2.0]),
    ]))
    .map(|(input, target): ([f32; 2], [f32; 1])| {
        Ok(SupervisedSample::new(
            Tensor::from_vec(input.to_vec(), &[2])?,
            Tensor::from_vec(target.to_vec(), &[1])?,
        ))
    })
    .build()?;
    let mut loader = DataLoader::builder(dataset)
        .collator(SupervisedStackCollator::new())
        .batch_size(2)
        .shuffle(false)
        .build()?;

    let mut model = LinearRegression::build_model(2, 1)?;
    let mut optimizer = SGD::new(0.01);
    for parameter in crate::legacy::trainer::TrainableModel::params(&model) {
        optimizer.register(parameter);
    }

    let result = Trainer::silent().supervised().fit(
        &mut model,
        &mut optimizer,
        &mut loader,
        EpochSchedule::new(2)?,
    )?;
    assert_eq!(result.units_completed, 2);
    assert!(result.final_loss.is_finite());
    Ok(())
}
