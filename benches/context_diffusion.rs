//! Product route benchmark. Run via scripts/benchmark_context_diffusion.py.
//! No test includes, copied models, or replay Trainer implementations.
#[path = "support/counting_allocator.rs"]
mod allocation;
#[cfg(feature = "benchmarkAlloc")]
#[global_allocator]
static ALLOCATOR: allocation::CountingAllocator = allocation::CountingAllocator;

use serde::Serialize;
use std::{cell::RefCell, rc::Rc, time::Instant};
use trench_deep::{
    nn::{Diffusion, DiffusionScheduler, Unet},
    optimizer::{Adam, Optimizer},
    runtime::prepared::{PreparedMode, PreparedModelExecutor},
    trainer::*,
    *,
};

#[derive(Serialize)]
struct Measurement {
    phase: &'static str,
    seconds: f64,
    memory: allocation::Snapshot,
    start_live_bytes: usize,
    peak_extra_bytes: usize,
}
fn measure<T>(phase: &'static str, f: impl FnOnce() -> MlResult<T>) -> MlResult<(T, Measurement)> {
    let before = allocation::begin();
    let start = Instant::now();
    let value = f()?;
    let seconds = start.elapsed().as_secs_f64();
    let memory = allocation::delta(before);
    Ok((
        value,
        Measurement {
            phase,
            seconds,
            memory,
            start_live_bytes: before.live_bytes,
            peak_extra_bytes: memory.peak_live_bytes.saturating_sub(before.live_bytes),
        },
    ))
}
fn context(route: ExecutionRoute) -> MlResult<ExecutionContext> {
    ExecutionContext::builder()
        .initialization_seed(7)
        .route(route)
        .build()
}
fn model(ctx: &ExecutionContext) -> MlResult<Diffusion> {
    Diffusion::new(
        ctx,
        Unet::new(ctx, 1, 8, &[1, 2], 4, &[])?,
        DiffusionScheduler::linear(10, 1e-4, 0.02)?,
        11,
    )
}
fn optimizer(ctx: &ExecutionContext, model: &Diffusion) -> MlResult<Adam> {
    let mut adam = Adam::new(ctx, 1e-3, 0.9, 0.999, 1e-8)?;
    adam.register_all(&model.parameters())?;
    Ok(adam)
}
fn noise_values(t: usize) -> Vec<f32> {
    (0..128)
        .map(|i| ((i + 137 * t) as f32 * 0.13).cos())
        .collect()
}
fn weights(model: &Diffusion) -> MlResult<Vec<f32>> {
    let mut values = Vec::new();
    for parameter in model.parameters() {
        values.extend(parameter.tensor().to_vec()?);
    }
    Ok(values)
}
#[derive(Serialize)]
struct Evidence {
    initial_weights: Vec<f32>,
    prediction: Vec<f32>,
    loss: f32,
    gradients: Vec<f32>,
    updated_weights: Vec<f32>,
}
fn stages(
    route: ExecutionRoute,
    evidence: bool,
    prepared: bool,
) -> MlResult<(Vec<Measurement>, Option<Evidence>)> {
    let mut measurements = Vec::with_capacity(8);
    let (ctx, m) = measure("context_init", || context(route))?;
    measurements.push(m);
    let (model, m) = measure("model_init", || model(&ctx))?;
    measurements.push(m);
    let (mut adam, m) = measure("optimizer_init_register", || optimizer(&ctx, &model))?;
    measurements.push(m);
    let image = ctx.tensor(vec![0.5; 128], &[2, 1, 8, 8])?.as_variable()?;
    let noise = ctx.tensor(noise_values(3), &[2, 1, 8, 8])?;
    let initial_weights = if evidence {
        weights(&model)?
    } else {
        Vec::new()
    };
    let inputs = if prepared {
        Some(
            model
                .training_feeds(&noise, 3)?
                .with("image", image.tensor().clone())?,
        )
    } else {
        None
    };
    let mut plan = if let Some(inputs) = &inputs {
        let (plan, m) = measure("prepare_training", || ctx.prepare_model(&model, inputs))?;
        measurements.push(m);
        Some(plan)
    } else {
        None
    };
    let before = allocation::begin();
    let start = Instant::now();
    let mut finish = |prediction: Variable, loss: Variable| -> MlResult<Option<Evidence>> {
        let seconds = start.elapsed().as_secs_f64();
        let memory = allocation::delta(before);
        measurements.push(Measurement {
            phase: "forward",
            seconds,
            memory,
            start_live_bytes: before.live_bytes,
            peak_extra_bytes: memory.peak_live_bytes.saturating_sub(before.live_bytes),
        });
        let (_, m) = measure("backward", || loss.backward())?;
        measurements.push(m);
        let mut result = if evidence {
            let mut gradients = Vec::new();
            for p in model.parameters() {
                gradients.extend_from_slice(p.grad()?.expect("parameter gradient").data());
            }
            Some(Evidence {
                initial_weights,
                prediction: prediction.tensor().to_vec()?,
                loss: loss.tensor().to_vec()?[0],
                gradients,
                updated_weights: Vec::new(),
            })
        } else {
            None
        };
        let (_, m) = measure("adam_step", || adam.step())?;
        measurements.push(m);
        if let Some(ref mut r) = result {
            r.updated_weights = weights(&model)?;
        }
        Ok(result)
    };
    let result = if let Some(plan) = &mut plan {
        plan.run(&model, inputs.as_ref().unwrap(), |out| {
            finish(out.prediction.unwrap(), out.loss)
        })?
    } else {
        ctx.with_training_scope(|| {
            let (prediction, loss) = model.forward_loss_with_noise(&image, &noise, 3)?;
            finish(prediction, loss)
        })?
    };
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    Ok((measurements, result))
}
struct LossObserver(Rc<RefCell<Vec<f32>>>);
impl TrainingObserver for LossObserver {
    fn on_batch_end(&mut self, event: &BatchEndContext) {
        self.0.borrow_mut().push(event.loss);
    }
}
fn training(route: ExecutionRoute, prepared: bool) -> MlResult<(Measurement, Vec<f32>, Vec<f32>)> {
    // Fresh initialization and empty Adam moments before each measured trajectory.
    let ctx = context(route)?;
    let mut model = model(&ctx)?;
    let mut adam = optimizer(&ctx, &model)?;
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
    let losses = Rc::new(RefCell::new(Vec::with_capacity(3)));
    let trainer = Trainer::builder()
        .metrics(Metrics::none())
        .show_progress(false)
        .build()
        .with_observer(Box::new(LossObserver(losses.clone())));
    let schedule = EpochSchedule::new(3)?.with_tolerance(1e-10);
    let (result, measurement) = measure("training_3_epochs", || {
        if prepared {
            trainer
                .prepared(&ctx)
                .fit(&mut model, &mut adam, &mut loader, schedule)
        } else {
            trainer
                .unsupervised(&ctx)
                .fit(&mut model, &mut adam, &mut loader, schedule)
        }
    })?;
    assert_eq!(result.units_completed, 3);
    let losses = losses.borrow().clone();
    assert_eq!(losses.len(), 3);
    assert!(losses.iter().all(|v| v.is_finite()));
    Ok((measurement, losses, weights(&model)?))
}
fn sampling(route: ExecutionRoute, prepared: bool) -> MlResult<(Vec<Measurement>, Vec<f32>)> {
    let ctx = context(route)?;
    let model = model(&ctx)?;
    let initial = ctx.tensor(
        (0..128).map(|i| (i as f32 * 0.17).sin()).collect(),
        &[2, 1, 8, 8],
    )?;
    let noises = (0..10)
        .map(|t| ctx.tensor(noise_values(t), &[2, 1, 8, 8]))
        .collect::<MlResult<Vec<_>>>()?;
    let mut measurements = Vec::new();
    let mut plans = if prepared {
        let (plans, m) = measure("prepare_sampling", || {
            [0, 1]
                .iter()
                .map(|&t| {
                    let inputs = model
                        .step_feeds(&noises[t], t)?
                        .with("image", initial.clone())?;
                    ctx.prepare_forward(
                        &inputs,
                        &model.parameters(),
                        PreparedMode::Inference,
                        |inputs| {
                            Ok(vec![
                                model.reverse_step_with_feeds(inputs.get("image")?, inputs)?,
                            ])
                        },
                    )?
                    .into_executor(&ctx)
                })
                .collect::<MlResult<Vec<_>>>()
        })?;
        measurements.push(m);
        plans
    } else {
        Vec::new()
    };
    let (output, measurement) = measure("sampling_10_steps", || {
        if !prepared {
            return model.sample_with_noise(&initial, &noises);
        }
        let mut image = initial.clone();
        for t in (0..10).rev() {
            let inputs = model.step_feeds(&noises[t], t)?.with("image", image)?;
            image =
                plans[usize::from(t != 0)]
                    .with_inputs(&inputs, &model.parameters(), |out| Ok(out[0].clone()))?;
        }
        Ok(image)
    })?;
    measurements.push(measurement);
    Ok((measurements, output.to_vec()?))
}

fn operation_cases(route: ExecutionRoute) -> MlResult<Vec<Measurement>> {
    use trench_deep::contracts::Operation;
    let ctx = context(route)?;
    let cases = [
        (
            "matmul_batch_broadcast",
            Operation::Matmul,
            vec![vec![2, 1, 8, 8], vec![1, 4, 8, 8]],
        ),
        (
            "conv2d",
            Operation::Conv2d {
                stride: (1, 1),
                padding: (1, 1),
            },
            vec![vec![2, 8, 8, 8], vec![8, 8, 3, 3], vec![8]],
        ),
        (
            "group_norm",
            Operation::GroupNorm {
                groups: 4,
                epsilon: 1e-5,
            },
            vec![vec![2, 8, 8, 8], vec![8], vec![8]],
        ),
        (
            "attention_transpose",
            Operation::Transpose(vec![0, 2, 1]),
            vec![vec![2, 8, 64]],
        ),
        (
            "attention_reshape",
            Operation::Reshape(vec![2, 8, 64]),
            vec![vec![2, 8, 8, 8]],
        ),
    ];
    let mut results = Vec::with_capacity(cases.len());
    for (name, operation, shapes) in cases {
        let inputs = shapes
            .iter()
            .map(|shape| {
                ctx.tensor(
                    (0..shape.iter().product())
                        .map(|i| (i as f32 * 0.17).sin())
                        .collect(),
                    shape,
                )
            })
            .collect::<MlResult<Vec<_>>>()?;
        let refs = inputs.iter().collect::<Vec<_>>();
        let (output, m) = measure(name, || ctx.no_grad(|| ctx.execute(&operation, &refs)))?;
        std::hint::black_box(output);
        results.push(m);
    }
    Ok(results)
}
fn verify_rng_feeds(route: ExecutionRoute) -> MlResult<serde_json::Value> {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let ctx = context(route)?;
    let mut model = model(&ctx)?;
    let initial = weights(&model)?;
    let image = ctx.tensor(vec![0.5; 128], &[2, 1, 8, 8])?.as_variable()?;
    let mut rng = StdRng::seed_from_u64(11);
    let mut feeds = Vec::new();
    // Validate actual product forward_loss consumption against explicit feeds;
    // this is outside all measurements and does not implement a Trainer loop.
    for _ in 0..3 {
        let values: Vec<f32> = (0..128)
            .map(|_| {
                (-2.0 * rng.random::<f32>().max(f32::MIN_POSITIVE).ln()).sqrt()
                    * (std::f32::consts::TAU * rng.random::<f32>()).cos()
            })
            .collect();
        let t = rng.random_range(0..10);
        let noise = ctx.tensor(values.clone(), &[2, 1, 8, 8])?;
        let actual = ctx.with_training_scope(|| {
            let (p, l) = model.forward_loss(&image)?;
            Ok((p.tensor().to_vec()?, l.tensor().to_vec()?))
        })?;
        let expected = ctx.with_training_scope(|| {
            let (p, l) = model.forward_loss_with_noise(&image, &noise, t)?;
            Ok((p.tensor().to_vec()?, l.tensor().to_vec()?))
        })?;
        assert_eq!(actual, expected);
        feeds.push(serde_json::json!({"t":t,"noise":values,"loss":actual.1}));
    }
    assert_eq!(weights(&model)?, initial);
    Ok(serde_json::json!(feeds))
}
#[derive(Serialize)]
struct MemoryPoint {
    phase: &'static str,
    iteration: usize,
    memory: allocation::Snapshot,
    tensors: usize,
    graph_nodes: usize,
    backward_nodes: usize,
    saved_references: usize,
}
fn memory_point(
    ctx: &ExecutionContext,
    phase: &'static str,
    iteration: usize,
) -> MlResult<MemoryPoint> {
    // Take allocator snapshot before querying graph statistics.
    let memory = allocation::snapshot();
    let graph = ctx.graph_stats()?;
    Ok(MemoryPoint {
        phase,
        iteration,
        memory,
        tensors: graph.tensors,
        graph_nodes: graph.graph_nodes,
        backward_nodes: graph.dynamic_backward_nodes,
        saved_references: graph.saved_tensor_references,
    })
}
fn memory_lifecycle(route: ExecutionRoute, prepared: bool) -> MlResult<serde_json::Value> {
    // Preallocate report storage so report growth cannot look like model leakage.
    let mut points = Vec::with_capacity(110);
    let before = allocation::snapshot();
    let ctx = context(route)?;
    let model = model(&ctx)?;
    let mut adam = optimizer(&ctx, &model)?;
    let image = ctx.tensor(vec![0.5; 128], &[2, 1, 8, 8])?.as_variable()?;
    let noise = ctx.tensor(noise_values(3), &[2, 1, 8, 8])?;
    let inputs = if prepared {
        Some(
            model
                .training_feeds(&noise, 3)?
                .with("image", image.tensor().clone())?,
        )
    } else {
        None
    };
    let mut plan: Option<PreparedModelExecutor> = inputs
        .as_ref()
        .map(|inputs| ctx.prepare_model(&model, inputs))
        .transpose()?;
    let arena_bytes = plan
        .as_ref()
        .map(|p| p.executor().arena_bytes())
        .unwrap_or(0);
    let workspace_bytes = plan
        .as_ref()
        .map(|p| p.executor().workspace_bytes())
        .unwrap_or(0);
    points.push(memory_point(&ctx, "initialized", 0)?);
    let resident_tensors = ctx.graph_stats()?.tensors;
    for iteration in 0..105 {
        allocation::begin();
        let mut finish = |prediction: Variable, loss: Variable| -> MlResult<()> {
            if iteration == 5 {
                points.push(memory_point(&ctx, "after_forward", iteration)?);
            }
            loss.backward()?;
            if iteration == 5 {
                points.push(memory_point(&ctx, "after_backward", iteration)?);
            }
            adam.step()?;
            if iteration == 5 {
                points.push(memory_point(&ctx, "after_adam", iteration)?);
            }
            std::hint::black_box(prediction);
            Ok(())
        };
        if let Some(plan) = &mut plan {
            plan.run(&model, inputs.as_ref().unwrap(), |out| {
                finish(out.prediction.unwrap(), out.loss)
            })?;
        } else {
            ctx.with_training_scope(|| {
                let (prediction, loss) = model.forward_loss_with_noise(&image, &noise, 3)?;
                finish(prediction, loss)
            })?;
        }
        if iteration >= 5 {
            let point = memory_point(&ctx, "after_scope", iteration - 5)?;
            assert_eq!(point.tensors, resident_tensors);
            assert_eq!(point.graph_nodes, 0);
            assert_eq!(point.saved_references, 0);
            points.push(point);
        }
    }
    drop((plan, inputs, noise, image, adam, model));
    points.push(memory_point(&ctx, "after_model_drop", 100)?);
    drop(ctx);
    let after = allocation::snapshot();
    Ok(
        serde_json::json!({"before":before, "points":points, "after_context_drop":after,
        "arena_bytes":arena_bytes,"workspace_bytes":workspace_bytes,"warmup":5, "batches":100, "report_storage_preallocated":true}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(feature = "debugging")
        || cfg!(feature = "enableVisualization")
        || cfg!(debug_assertions)
    {
        return Err("use release with legacyBenchmark; no debugging/visualization".into());
    }
    let args: Vec<_> = std::env::args().collect();
    let route = match args.get(1).map(String::as_str) {
        Some("p1" | "prepared") => ExecutionRoute::P1,
        Some("legacy") => ExecutionRoute::Legacy,
        _ => return Err("expected p1|prepared|legacy output.json [samples]".into()),
    };
    let prepared = args[1] == "prepared";
    let output_path = args.get(2).ok_or("expected output path")?;
    let samples: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(30);
    if samples == 0 {
        return Err("samples must be positive".into());
    }
    let warmup = 5;
    let mut measurements = Vec::with_capacity(samples * 13);
    // Evidence pass is outside timing and memory statistics.
    let (_, evidence) = stages(route, true, prepared)?;
    let rng_feeds = verify_rng_feeds(route)?;
    let mut train_losses = Vec::new();
    let mut train_weights = Vec::new();
    let mut sample_output = Vec::new();
    for iteration in 0..warmup + samples {
        let (stages, _) = stages(route, false, prepared)?;
        let (train, losses, trained) = training(route, prepared)?;
        let (sample, output) = sampling(route, prepared)?;
        let operations = operation_cases(route)?;
        if iteration >= warmup {
            measurements.extend(stages);
            measurements.push(train);
            measurements.extend(sample);
            measurements.extend(operations);
        }
        if iteration == 0 {
            train_losses = losses;
            train_weights = trained;
            sample_output = output;
        } else {
            assert_eq!(losses, train_losses);
            assert_eq!(trained, train_weights);
            assert_eq!(output, sample_output);
        }
    }
    let memory = if cfg!(feature = "benchmarkAlloc") {
        Some(memory_lifecycle(route, prepared)?)
    } else {
        None
    };
    let result = serde_json::json!({"schema":1, "route":args[1], "instrumented":cfg!(feature="benchmarkAlloc"),
        "available_parallelism":std::thread::available_parallelism()?.get(),
        "warmup":warmup, "samples":samples, "batch":2, "measurements":measurements,
        "evidence":evidence, "rng_feeds":rng_feeds, "training_losses":train_losses, "training_weights":train_weights,
        "sampling_output":sample_output, "memory_lifecycle":memory,
        "fixture":{"shape":[2,1,8,8],"dim":8,"multipliers":[1,2],"groups":4,"middle_attention":true,
        "steps":10,"beta_start":1e-4,"beta_end":0.02,"weight_seed":7,"rng_seed":11,"explicit_t":3},
        "notes":["stage restoration/setup excluded; fresh model and Adam per repetition",
        "training includes loader iteration, common Trainer, RNG, Adam, scope cleanup and loss observer",
        "sampling output conversion excluded; allocator requested bytes are not RSS",
        "instrumented timing must not be used as ordinary performance", "operation microcases remain eager diagnostics on every route", "prepared forward includes scope entry and exported ownership copies; backward includes gradient publication", "prepared training_3_epochs includes first preparation; sampling preparation reported separately"]});
    std::fs::write(output_path, serde_json::to_vec(&result)?)?;
    Ok(())
}
