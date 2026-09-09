#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use old::{
    nn::Parameter as OldParameter, optimizer::Optimizer as OldOptimizer, tensor::TensorBase,
    trainer::TrainableModel as OldModel,
};
use trench_deep::legacy as old;
use trench_deep::{
    nn::LinearRegression,
    optimizer::{Optimizer, SGD},
    trainer::*,
    *,
};

fn close(a: f32, b: f32) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3f32.max(1e-3 * a.abs().max(b.abs())),
        "{a} versus {b}"
    );
}

pub fn run_case(iterations: usize) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use std::time::Instant;
    if iterations == 0 {
        return Err("at least one iteration is required".into());
    }
    let initialization = Instant::now();
    let ctx = ExecutionContext::new();
    let mut model = LinearRegression::new(&ctx, 2, 1)?;
    let public_model_initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let legacy_initialization = Instant::now();
    let mut baseline = old::comparison::linear::LinearRegression::build_model(2, 1)?;
    let legacy_model_initialization_ms = legacy_initialization.elapsed().as_secs_f64() * 1000.0;
    for (p, q) in model.parameters().iter().zip(baseline.params()) {
        let shape = p.tensor().shape()?;
        let values = vec![0.1; p.tensor().numel()?];
        ctx.replace_parameter(
            p.variable(),
            TensorBuffer::from_vec(values.clone(), &shape)?,
        )?;
        q.tensor()
            .replace(old::tensor::GlobalTensor::from_vec(values, &shape)?);
    }
    let mut samples = Vec::new();
    let mut old_samples = Vec::new();
    for i in 0..8 {
        let x = vec![i as f32 / 8.0, 1.0 - i as f32 / 8.0];
        let y = vec![x[0] * 0.3 + x[1] * 0.7];
        samples.push(SupervisedSample::new(
            ctx.tensor(x.clone(), &[2])?,
            ctx.tensor(y.clone(), &[1])?,
        ));
        old_samples.push(old::trainer::SupervisedSample::new(
            old::tensor::Tensor::from_vec(x, &[2])?,
            old::tensor::Tensor::from_vec(y, &[1])?,
        ));
    }
    let mut loader = DataLoader::builder(InMemoryDataset::new(samples)?)
        .collator(SupervisedStackCollator::new())
        .batch_size(2)
        .shuffle(false)
        .build()?;
    let mut old_loader =
        old::trainer::DataLoader::builder(old::trainer::InMemoryDataset::new(old_samples)?)
            .collator(old::trainer::SupervisedStackCollator::new())
            .batch_size(2)
            .shuffle(false)
            .build()?;
    let mut optimizer = SGD::new(&ctx, 0.01)?;
    optimizer.register_all(&model.parameters())?;
    let mut old_optimizer = old::optimizer::SGD::new(0.01);
    for p in baseline.params() {
        old_optimizer.register(p);
    }
    let trainer = SupervisedTrainer::silent(&ctx);
    let old_trainer = old::trainer::SupervisedTrainer::silent();
    let initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let mut public = Vec::new();
    let mut legacy = Vec::new();
    for i in 0..=iterations {
        let start = Instant::now();
        let actual = trainer.fit(
            &mut model,
            &mut optimizer,
            &mut loader,
            EpochSchedule::new(1)?,
        )?;
        let public_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let expected = old_trainer.fit(
            &mut baseline,
            &mut old_optimizer,
            &mut old_loader,
            old::trainer::EpochSchedule::new(1)?,
        )?;
        let legacy_ms = start.elapsed().as_secs_f64() * 1000.0;
        close(actual.final_loss, expected.final_loss);
        for (p, q) in model.parameters().iter().zip(baseline.params()) {
            for (a, b) in p.tensor().to_vec()?.into_iter().zip(q.tensor().data()) {
                close(a, *b);
            }
        }
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        if i != 0 {
            public.push(public_ms);
            legacy.push(legacy_ms);
        }
    }
    let summary = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let n = values.len();
        let median = values[n / 2];
        serde_json::json!({"median_ms":median,"p95_ms":values[((n as f64*0.95).ceil() as usize)-1],"samples_per_second":8000.0/median,"samples_ms":values})
    };
    let (legacy_storage_handles, legacy_graph_nodes) = old::comparison::statistics()?;
    Ok(
        serde_json::json!({"legacy_storage_handles":legacy_storage_handles,"legacy_graph_nodes":legacy_graph_nodes,"case":"linear_trainer_loader_to_optimizer","iterations":iterations,"initialization_ms":initialization_ms,"public_model_initialization_ms":public_model_initialization_ms,"legacy_model_initialization_ms":legacy_model_initialization_ms,"public":summary(public),"legacy":summary(legacy),"parity":"passed","batches_per_iteration":4,"public_graph_nodes":ctx.graph_stats()?.graph_nodes,"public_storage_handles":ctx.graph_stats()?.tensors}),
    )
}
