#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use old::tensor::{TensorBase, operators::Function};
use trench_deep::{ExecutionContext, legacy as old};

pub fn run_case(iterations: usize) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use std::{hint::black_box, time::Instant};
    if iterations == 0 {
        return Err("at least one iteration is required".into());
    }
    let initialization = Instant::now();
    let ctx = ExecutionContext::new();
    let values = (0..256).map(|i| (i % 17) as f32 * 0.01).collect::<Vec<_>>();
    let x = ctx.tensor(values.clone(), &[16, 16])?;
    let y = ctx.tensor(values.clone(), &[16, 16])?;
    let public_model_initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let legacy_initialization = Instant::now();
    let ox = old::tensor::Tensor::from_vec(values.clone(), &[16, 16])?;
    let oy = old::tensor::Tensor::from_vec(values, &[16, 16])?;
    let operation = old::tensor::operators::Matmul::new()?;
    let legacy_model_initialization_ms = legacy_initialization.elapsed().as_secs_f64() * 1000.0;
    let initialization_ms = initialization.elapsed().as_secs_f64() * 1000.0;
    let actual = x.matmul(&y)?.to_vec()?;
    let expected = operation
        .forward(&[&ox, &oy])?
        .into_iter()
        .next()
        .ok_or("missing output")?;
    for (a, b) in actual.iter().zip(expected.data()) {
        assert!((*a - *b).abs() <= 1e-3f32.max(1e-3 * a.abs().max(b.abs())));
    }
    let mut public = Vec::new();
    let mut legacy = Vec::new();
    for i in 0..=iterations {
        let start = Instant::now();
        let result = black_box(x.matmul(black_box(&y))?);
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        drop(result);
        let start = Instant::now();
        let result = black_box(operation.forward(black_box(&[&ox, &oy]))?);
        let old_elapsed = start.elapsed().as_secs_f64() * 1e6;
        drop(result);
        if i != 0 {
            public.push(elapsed);
            legacy.push(old_elapsed);
        }
    }
    let summary = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let n = values.len();
        let median = values[n / 2];
        serde_json::json!({"median_us":median,"p95_us":values[((n as f64*0.95).ceil() as usize)-1],"operations_per_second":1e6/median,"samples_us":values})
    };
    let (legacy_storage_handles, legacy_graph_nodes) = old::comparison::statistics()?;
    Ok(
        serde_json::json!({"legacy_storage_handles":legacy_storage_handles,"legacy_graph_nodes":legacy_graph_nodes,"case":"matmul_16x16_forward","iterations":iterations,"initialization_ms":initialization_ms,"public_model_initialization_ms":public_model_initialization_ms,"legacy_model_initialization_ms":legacy_model_initialization_ms,"public":summary(public),"legacy":summary(legacy),"parity":"passed","public_storage_handles":ctx.graph_stats()?.tensors,"public_graph_nodes":ctx.graph_stats()?.graph_nodes}),
    )
}
