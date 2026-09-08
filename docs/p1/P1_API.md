# Explicit execution and provider contracts

```rust
use trench_deep::{ExecutionContext, MlResult};

fn example() -> MlResult<()> {
    let ctx = ExecutionContext::builder().initialization_seed(7).model_seed(11).build();
    let weight = ctx.parameter(vec![2.0], &[])?;
    let loss = weight.square()?;
    loss.backward()?; // enableBackward, or an injected autograd implementation
    let gradient = weight.grad()?;
    assert_eq!(gradient.map(|g| g.data()[0]), Some(4.0));
    Ok(())
}
```

`ParameterId` is independent of graph nodes. Register shared parameters once with
`Optimizer::register_all`; trainer validation compares unique model and optimizer
IDs. `Parameter::tensor()` keeps tracking. `detach()` shares the value buffer but
removes tracking. Tensor handles do not keep a dropped context alive.

`contracts::{TensorStore, AutogradEngine, OperationProvider}` own storage, graph
representation/traversal, and operations respectively. Inject implementations via
`ExecutionContextBuilder::empty().storage(...).autograd(...).operations(...).build()`.
Configuration is fixed after construction. A provider is responsible for its trait
contract, including atomic errors and idempotent removal. Custom operations receive
immutable views and return owned buffers/VJPs; they receive no storage mutation API.
`tests/support/mod.rs` contains independent slot storage, a vector tape, and reference
math used to run the same public model and trainer with all builtins removed.

| Feature | Implementation enabled |
|---|---|
| `builtinStorage` (default) | Existing backend path's CPU tensor store |
| `builtinKernels` (default) | Existing `backend/cpu` operation provider |
| `enableBackward` | Default reverse-mode tape |
| `enableVisualization` | Explicit context graph capture |
| `debugging` | Operation/backward tracing spans |
| `legacyBenchmark` | Preserved, separate legacy crate and comparison bridges |

Contracts and request APIs remain available without these features. Missing
implementations return `DependencyUnavailable`; unsupported provider operations
return `UnsupportedCapability`. Ordinary inference does not require autograd.
Capture requests without the feature fail explicitly before consuming the graph.

Layers implement `Layer::forward` once; default `predict` runs it under no-grad.
Four epoch trainers share validation, loading scopes, backward, clipping, hooks,
updates, cleanup, and successful completion events. RL retains its rollout loop and
uses the same training step. Default policy prediction runs `policy_logits` under
no-grad. Scope cleanup errors are combined with the primary error; optimizer updates
are not rolled back after partial failure.

Initialization and model-noise streams belong to the context. Trainer data order
and RL actions use independent streams. `with_seed` on a trainer never reinitializes
parameters. `Diffusion::new` and `TwoArmedBandit::with_seed` own separate noise seeds.

`nn::{Unet, Diffusion, DiffusionScheduler}` include time embedding, residual and
attention blocks, down/up paths, linear/cosine schedules, training, and sampling.
Use `forward_loss_with_noise` and `sample_with_noise` for reproducible fixtures.
`Unet::with_dimensions` supports separate initial/output channels; DDPM requires
output channels to match the image. Checkpoint containers retain `.tdw`/JSON formats.
Use `fit_checkpointed` for interrupt saves. Complete resume and optimizer snapshots
are P2 capabilities and return explicit unsupported errors.

`ctx.graph_snapshot` captures the current graph. `ctx.backward_snapshot` captures
intermediate gradients before normal cleanup. Trainer observers receive owned
snapshots after successful scope cleanup, followed by the batch success event.
Snapshots can be written through `FileSnapshotWriter` or converted by `DotEncoder`.
