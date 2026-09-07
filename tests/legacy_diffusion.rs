#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use old::nn::{Layer as OldLayer, Parameter as OldParameter};
use old::tensor::{AutogradFunction, TensorBase, operators::Function};
use trench_deep::legacy as old;
use trench_deep::{nn::Unet, trainer::TrainableModel, *};

fn close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}");
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 1e-3f32.max(1e-3 * a.abs().max(b.abs())),
            "{label}[{i}]: {a} vs {b}"
        );
    }
}

pub fn run_case(repetitions: usize) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if repetitions == 0 {
        return Err("benchmark requires at least one iteration".into());
    }
    use old::optimizer::Optimizer as OldOptimizer;
    use std::time::Instant;
    use trench_deep::optimizer::Optimizer;
    let initialization = Instant::now();
    let ctx = ExecutionContext::new();
    let model = Unet::new(&ctx, 1, 2, &[1, 2], 1, &[0, 1])?;
    let public_model_initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let legacy_initialization = Instant::now();
    let mut baseline =
        old::comparison::unet::Unet::new(2, None, None, &[1, 2], 1, 1, &[true, true])?;
    let legacy_model_initialization_ms = legacy_initialization.elapsed().as_secs_f64() * 1000.0;
    let parameters = model.parameters();
    let legacy_parameters = baseline.params();
    assert_eq!(parameters.len(), legacy_parameters.len());
    for (i, (p, q)) in parameters.iter().zip(&legacy_parameters).enumerate() {
        let shape = p.tensor().shape()?;
        assert_eq!(shape, q.tensor().shape(), "parameter {i}");
        let values = (0..p.tensor().numel()?)
            .map(|j| {
                if shape.len() == 1 {
                    0.1 + (j % 3) as f32 * 0.01
                } else {
                    ((i * 13 + j * 7) % 19) as f32 * 0.003 - 0.027
                }
            })
            .collect::<Vec<_>>();
        ctx.replace_parameter(
            p.variable(),
            TensorBuffer::from_vec(values.clone(), &shape)?,
        )?;
        q.tensor()
            .replace(old::tensor::GlobalTensor::from_vec(values, &shape)?);
    }
    drop(legacy_parameters);
    let values = (0..16).map(|i| i as f32 * 0.03).collect::<Vec<_>>();
    let x = ctx.input(values.clone(), &[1, 1, 4, 4])?;
    let t = ctx.tensor(vec![0.5], &[1, 1])?;
    let ox = old::nn::Variable::new(old::tensor::Tensor::from_vec(values, &[1, 1, 4, 4])?);
    let ot = old::nn::Variable::new(old::tensor::Tensor::from_vec(vec![0.5], &[1, 1])?);
    let target = ctx.tensor(vec![0.05; 16], &[1, 1, 4, 4])?;
    let old_target = old::nn::Variable::new(old::tensor::Tensor::from_vec(
        vec![0.05; 16],
        &[1, 1, 4, 4],
    )?);
    let mut optimizer = trench_deep::optimizer::SGD::new(&ctx, 0.001)?;
    optimizer.register_all(&parameters)?;
    let mut old_optimizer = old::optimizer::SGD::new(0.001);
    for p in baseline.params() {
        old_optimizer.register(p);
    }
    let initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let mut public_times = Vec::new();
    let mut legacy_times = Vec::new();
    for _ in 0..repetitions + 1 {
        let start = Instant::now();
        let result = model.forward(&x, &t)?;
        let loss = result.mse_loss(&target, Reduction::Mean)?;
        loss.backward()?;
        let public_elapsed = start.elapsed();
        let start = Instant::now();
        let expected = baseline.forward_with_t(&ox, &ot)?;
        let old_loss = old::loss::MeanSquaredError::new()?.apply(&[&expected, &old_target])?;
        old_loss.backward()?;
        let legacy_elapsed = start.elapsed();
        close(
            &result.tensor().to_vec()?,
            expected.tensor().data(),
            "forward",
        );
        close(&loss.tensor().to_vec()?, old_loss.tensor().data(), "loss");
        for (i, (p, q)) in parameters.iter().zip(baseline.params()).enumerate() {
            let gradient = p.grad()?.ok_or("missing gradient")?;
            close(gradient.data(), q.grad().data(), &format!("gradient {i}"));
        }
        let start = Instant::now();
        optimizer.step()?;
        optimizer.zero_grad()?;
        let public_elapsed = public_elapsed + start.elapsed();
        let start = Instant::now();
        old_optimizer.step()?;
        old_optimizer.zero_grad()?;
        old::comparison::clear_graph();
        let legacy_elapsed = legacy_elapsed + start.elapsed();
        for (i, (p, q)) in parameters.iter().zip(baseline.params()).enumerate() {
            close(
                &p.tensor().to_vec()?,
                q.tensor().data(),
                &format!("updated parameter {i}"),
            );
        }
        public_times.push(public_elapsed.as_secs_f64() * 1000.0);
        legacy_times.push(legacy_elapsed.as_secs_f64() * 1000.0);
    }
    public_times.remove(0);
    legacy_times.remove(0);
    let summary = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let n = values.len();
        let median = values[n / 2];
        let p95 = values[((n as f64 * 0.95).ceil() as usize).saturating_sub(1)];
        serde_json::json!({"median_ms":median,"p95_ms":p95,"steps_per_second":1000.0/median,"samples_ms":values})
    };
    let (legacy_storage_handles, legacy_graph_nodes) = old::comparison::statistics()?;
    Ok(
        serde_json::json!({"legacy_storage_handles":legacy_storage_handles,"legacy_graph_nodes":legacy_graph_nodes,"case":"unet_forward_backward_sgd","initialization_ms":initialization_ms,"public_model_initialization_ms":public_model_initialization_ms,"legacy_model_initialization_ms":legacy_model_initialization_ms,"iterations":repetitions,
        "public":summary(public_times),"legacy":summary(legacy_times),"public_storage_handles":ctx.graph_stats()?.tensors,
        "public_graph_nodes":ctx.graph_stats()?.graph_nodes,"parity":"passed","tolerance":"atol=1e-3, rtol=1e-3"}),
    )
}

#[test]
fn unet_forward_loss_gradients_and_sgd_match_legacy() -> Result<(), Box<dyn std::error::Error>> {
    run_case(1)?;
    Ok(())
}
