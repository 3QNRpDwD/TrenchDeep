#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use trench_deep::nn::{Diffusion, DiffusionScheduler, Unet};
use trench_deep::optimizer::{Optimizer, SGD};
use trench_deep::trainer::*;
use trench_deep::*;

/// The configuration and loader pipeline of the preserved
/// diffusion_train_with_trainer, using the existing public P1 model unchanged.
#[test]
fn reference_diffusion_configuration_trains_with_adam_and_loader() -> MlResult<()> {
    use trench_deep::optimizer::Adam;
    let ctx = ExecutionContext::builder().initialization_seed(7).build();
    let unet = Unet::new(&ctx, 1, 8, &[1, 2], 4, &[])?;
    let mut model = Diffusion::new(&ctx, unet, DiffusionScheduler::linear(10, 1e-4, 0.02)?, 11)?;
    let mut optimizer = Adam::new(&ctx, 1e-3, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let dataset = DatasetBuilder::from_source(MemorySource::new(vec![
        ctx.tensor(vec![0.5; 64], &[1, 8, 8])?,
        ctx.tensor(vec![0.5; 64], &[1, 8, 8])?,
    ]))
    .map(|input| Ok(UnsupervisedSample::new(input)))
    .build()?;
    let mut loader = DataLoader::builder(dataset)
        .collator(UnsupervisedStackCollator::new())
        .batch_size(2)
        .shuffle(false)
        .build()?;
    let result = Trainer::builder()
        .metrics(Metrics::all())
        .show_progress(false)
        .build()
        .unsupervised(&ctx)
        .fit(
            &mut model,
            &mut optimizer,
            &mut loader,
            EpochSchedule::new(3)?.with_tolerance(1e-10),
        )?;
    assert_eq!(result.units_completed, 3);
    assert!(result.final_loss.is_finite());
    for key in [
        "avg_loss",
        "grad_norm",
        "update_ratio",
        "forward_secs",
        "backward_secs",
    ] {
        assert!(
            result
                .metrics
                .get(key)
                .is_some_and(|value| value.is_finite()),
            "missing or invalid metric: {key}"
        );
    }
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    for parameter in model.parameters() {
        assert!(parameter.grad()?.is_none());
    }
    Ok(())
}

#[test]
fn full_unet_trains_and_samples_through_public_api() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let unet = Unet::new(&ctx, 1, 2, &[1, 2], 1, &[0, 1])?;
    let mut model = Diffusion::new(&ctx, unet, DiffusionScheduler::linear(2, 0.001, 0.02)?, 7)?;
    let image = ctx.input((0..16).map(|i| i as f32 / 16.0).collect(), &[1, 1, 4, 4])?;
    let noise = ctx.tensor(vec![0.1; 16], &[1, 1, 4, 4])?;
    let (prediction, loss) = model.forward_loss_with_noise(&image, &noise, 1)?;
    assert_eq!(prediction.tensor().shape()?, vec![1, 1, 4, 4]);
    assert!(loss.tensor().item()?.is_finite());
    loss.backward()?;
    assert!(
        model
            .parameters()
            .iter()
            .all(|p| p.grad().is_ok_and(|g| g.is_some()))
    );
    drop(prediction);
    drop(loss);
    let mut optimizer = SGD::new(&ctx, 0.001)?;
    optimizer.register_all(&model.parameters())?;
    let images = [&image];
    let dataset = UnsupervisedDataset::new(&ctx, &images)?;
    UnsupervisedTrainer::silent(&ctx).fit(
        &mut model,
        &mut optimizer,
        &dataset,
        EpochSchedule::new(1)?,
    )?;
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    let sample = model.sample(&[1, 1, 4, 4])?;
    assert_eq!(sample.shape()?, vec![1, 1, 4, 4]);
    assert!(sample.to_vec()?.iter().all(|x| x.is_finite()));
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    Ok(())
}

#[test]
fn scheduler_matches_closed_form_and_t0_ignores_noise() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let scheduler = DiffusionScheduler::from_betas(vec![0.1, 0.2])?;
    let x = ctx.scalar(2.0)?;
    let noise = ctx.scalar(0.5)?;
    assert!(
        (scheduler.q_sample(&x, &noise, 1)?.item()?
            - (0.72f32.sqrt() * 2.0 + 0.28f32.sqrt() * 0.5))
            .abs()
            < 1e-6
    );
    let a = scheduler.reverse_step(&x, &noise, &ctx.scalar(1.0)?, 0)?;
    let b = scheduler.reverse_step(&x, &noise, &ctx.scalar(999.0)?, 0)?;
    assert_eq!(a.item()?, b.item()?);
    Ok(())
}

#[test]
fn diffusion_checkpoint_round_trip_preserves_sampling() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::new();
    let make = || {
        Diffusion::new(
            &ctx,
            Unet::new(&ctx, 1, 2, &[1], 1, &[])?,
            DiffusionScheduler::cosine(2, 0.008)?,
            7,
        )
    };
    let original = make()?;
    let mut restored = make()?;
    let path = std::env::temp_dir().join(format!(
        "trench-ddpm-{}-{}.tdw",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    original.save_checkpoint(&path)?;
    restored.load_checkpoint(&path)?;
    std::fs::remove_file(&path)?;
    let initial = ctx.tensor(vec![0.1; 16], &[1, 1, 4, 4])?;
    let noises = vec![ctx.tensor(vec![0.0; 16], &[1, 1, 4, 4])?; 2];
    assert_eq!(
        original.sample_with_noise(&initial, &noises)?.to_vec()?,
        restored.sample_with_noise(&initial, &noises)?.to_vec()?
    );
    Ok(())
}
