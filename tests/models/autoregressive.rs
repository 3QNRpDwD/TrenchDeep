use trench_deep::{nn::*, trainer::TrainableModel, ExecutionContext, MlResult, Variable};
use trench_deep::optimizer::{Adam, Optimizer};
use trench_deep::trainer::{
    AutoregressiveDataset, AutoregressiveSample, AutoregressiveStackCollator,
    AutoregressiveTrainer, DataLoader, EpochSchedule, InMemoryDataset,
};

fn sequence(context: &ExecutionContext, tokens: &[usize], vocab: usize) -> MlResult<Variable> {
    let mut data = vec![0.0; tokens.len() * vocab];
    for (row, token) in tokens.iter().copied().enumerate() {
        data[row * vocab + token] = 1.0;
    }
    context.input(data, &[tokens.len(), vocab])
}

#[test]
fn bigram_pilot_trains_end_to_end() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = BigramLm::new(&context, 4)?;
    let samples = [
        sequence(&context, &[0, 1, 2, 3, 0], 4)?,
        sequence(&context, &[1, 2, 3, 0, 1], 4)?,
    ];
    let refs = samples.iter().collect::<Vec<_>>();
    let dataset = AutoregressiveDataset::new(&context, &refs)?;
    let mut optimizer = Adam::new(&context, 0.05, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let result = AutoregressiveTrainer::silent(&context).fit(
        &mut model,
        &mut optimizer,
        &dataset,
        EpochSchedule::new(5)?.with_tolerance(0.0),
    )?;
    assert!(result.final_loss.is_finite());
    assert_eq!(context.graph_stats()?.graph_nodes, 0);
    Ok(())
}

#[test]
fn bigram_pilot_accepts_stacked_loader_batches() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = BigramLm::new(&context, 4)?;
    let samples = [
        sequence(&context, &[0, 1, 2, 3, 0], 4)?,
        sequence(&context, &[1, 2, 3, 0, 1], 4)?,
    ];
    let refs = samples.iter().collect::<Vec<_>>();
    let dataset = AutoregressiveDataset::new(&context, &refs)?;
    let mut loader = DataLoader::builder(InMemoryDataset::new(
        dataset
            .sequences
            .iter()
            .map(|v| AutoregressiveSample::new(v.tensor().clone()))
            .collect(),
    )?)
    .collator(AutoregressiveStackCollator::new())
    .batch_size(2)
    .shuffle(false)
    .build()?;
    let mut optimizer = Adam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let result = AutoregressiveTrainer::silent(&context).fit_loader(
        &mut model,
        &mut optimizer,
        &mut loader,
        EpochSchedule::new(2)?.with_tolerance(0.0),
    )?;
    assert!(result.final_loss.is_finite());
    assert_eq!(context.graph_stats()?.graph_nodes, 0);
    Ok(())
}

#[test]
fn autoregressive_padding_is_rejected_until_it_has_loss_semantics() -> MlResult<()> {
    let context = ExecutionContext::new();
    let mut model = BigramLm::new(&context, 4)?;
    let sample = sequence(&context, &[0, 1, 2], 4)?;
    let refs = [&sample];
    let dataset = AutoregressiveDataset::new(&context, &refs)?.with_pad_token_id(0);
    let mut optimizer = Adam::new(&context, 0.02, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    assert!(
        AutoregressiveTrainer::silent(&context)
            .fit(&mut model, &mut optimizer, &dataset, EpochSchedule::new(1)?)
            .is_err()
    );
    Ok(())
}
