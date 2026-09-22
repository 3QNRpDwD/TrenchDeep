//! Deterministic prepared-only MLP benchmark: cargo bench --bench prepared_mlp.
#[path = "support/counting_allocator.rs"]
mod allocation;
#[cfg(feature = "benchmarkAlloc")]
#[global_allocator]
static ALLOCATOR: allocation::CountingAllocator = allocation::CountingAllocator;
use std::time::Instant;
use trench_deep::{
    nn::{Activation, ActivationKind, Layer, Linear, Sequential},
    optimizer::{Adam, Optimizer},
    runtime::prepared::*,
    trainer::*,
    *,
};

struct Model {
    ctx: ExecutionContext,
    net: Sequential,
}
impl TrainableModel for Model {
    fn context_id(&self) -> ContextId {
        self.ctx.id()
    }
    fn parameters(&self) -> Vec<&Parameter> {
        self.net.parameters()
    }
}

fn main() -> MlResult<()> {
    let args: Vec<_> = std::env::args().collect();
    let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let width: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
    let ctx = ExecutionContext::builder().initialization_seed(7).build();
    let mut net = Sequential::new(&ctx, "mlp");
    for i in 0..3 {
        net.push(Box::new(Linear::new(&ctx, width, width, format!("l{i}"))?))?;
        net.push(Box::new(Activation::new(
            &ctx,
            ActivationKind::ReLU,
            format!("a{i}"),
        )))?;
    }
    let mut model = Model {
        ctx: ctx.clone(),
        net,
    };
    let x = ctx.input(
        (0..batch * width)
            .map(|i| (i as f32 * 0.01).sin())
            .collect(),
        &[batch, width],
    )?;
    let target = ctx.tensor(vec![0.1; batch * width], &[batch, width])?;
    let inputs = ExecutionInputs::new("mlp")
        .with("x", x.tensor().clone())?
        .with("target", target)?;
    let mut optimizer = Adam::new(&ctx, 0.001, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;
    let start = Instant::now();
    let mut prepared = ctx.prepare_objective(&mut model, &objective(), &inputs)?;
    println!("prepare_ms={}", start.elapsed().as_secs_f64() * 1000.0);
    for step in 0..steps {
        let memory = allocation::begin();
        let start = Instant::now();
        let mut forward = 0.0;
        let mut backward = 0.0;
        let mut adam = 0.0;
        let mut loss = 0.0;
        prepared.run(&model, &inputs, |output| {
            forward = start.elapsed().as_secs_f64() * 1000.0;
            loss = output.loss.tensor().item()?;
            let t = Instant::now();
            output.loss.backward()?;
            backward = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            optimizer.step()?;
            adam = t.elapsed().as_secs_f64() * 1000.0;
            Ok(())
        })?;
        println!(
            "{step},{forward},{backward},{adam},{},{loss}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        let stats = ctx.graph_stats()?;
        assert_eq!(stats.graph_nodes, 0);
        assert_eq!(stats.dynamic_backward_nodes, 0);
        assert_eq!(stats.saved_tensor_references, 0);
        #[cfg(feature = "benchmarkAlloc")]
        println!(
            "memory,{step},{}",
            serde_json::to_string(&allocation::delta(memory)).unwrap()
        );
        #[cfg(not(feature = "benchmarkAlloc"))]
        let _ = memory;
    }
    let mut hash = 0xcbf29ce484222325u64;
    for parameter in model.parameters() {
        for value in parameter.tensor().to_vec()? {
            hash ^= value.to_bits() as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    println!("weights_fnv64={hash:016x}");
    Ok(())
}



impl trench_deep::trainer::ForwardModel for Model {
    fn forward(&self, input: &trench_deep::Variable) -> MlResult<trench_deep::Variable> {
        
        self.net.apply(input)
    }
}

fn objective() -> trench_deep::trainer::Supervised<trench_deep::loss::MseLoss> {
    trench_deep::trainer::Supervised::new(trench_deep::loss::MseLoss::new(trench_deep::Reduction::Mean))
}
