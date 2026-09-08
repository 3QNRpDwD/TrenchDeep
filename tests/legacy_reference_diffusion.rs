#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use old::{
    nn::{Layer, Parameter as OldParameter},
    optimizer::Optimizer as OldOptimizer,
    tensor::TensorBase,
};
use trench_deep::{
    legacy as old,
    nn::{Diffusion, DiffusionScheduler, Unet},
    optimizer::{Adam, Optimizer},
    trainer::TrainableModel,
    *,
};
#[path = "../examples/replay_reference_diffusion.rs"]
mod replay;

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 1e-3f32.max(1e-3 * a.abs().max(b.abs())),
            "{label}[{index}]: {a} != {b}"
        );
    }
}

/// Direct, unmodified legacy DDPM forward followed by replay in the existing P1
/// model. This is a numerical baseline, not the future unified Trainer adapter.
#[test]
fn original_reference_ddpm_draws_replay_through_context_and_adam()
-> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::new();
    let model = Diffusion::new(
        &ctx,
        Unet::new(&ctx, 1, 8, &[1, 2], 4, &[])?,
        DiffusionScheduler::linear(10, 1e-4, 0.02)?,
        0,
    )?;
    let mut baseline = old::comparison::ReferenceDiffusion::new(
        1,
        8,
        8,
        &[1, 2],
        4,
        &[false, false],
        10,
        1e-4,
        0.02,
    )?;
    let parameters = model.parameters();
    assert_eq!(parameters.len(), baseline.unet.params().len());
    let named_entries = model.unet.named_parameters();
    let old_descriptors = replay::mapping::describe(
        baseline
            .unet
            .comparison_named_parameters()
            .into_iter()
            .map(|(name, p)| (name, p.tensor().shape().to_vec(), p.node_id()))
            .collect(),
    )?;
    let descriptors = replay::mapping::describe(
        named_entries
            .iter()
            .map(|(name, p)| Ok((name.clone(), p.tensor().shape()?, p.id())))
            .collect::<MlResult<Vec<_>>>()?,
    )?;
    let named: std::collections::BTreeMap<_, _> = named_entries.into_iter().collect();
    assert_eq!(
        named
            .values()
            .map(|p| p.id())
            .collect::<std::collections::HashSet<_>>(),
        parameters
            .iter()
            .map(|p| p.id())
            .collect::<std::collections::HashSet<_>>()
    );
    replay::mapping::validate(&descriptors, &old_descriptors)?;
    assert_eq!(
        descriptors.len(),
        parameters.len(),
        "named enumeration must cover every parameter"
    );
    assert_eq!(old_descriptors.len(), baseline.unet.params().len());
    let old_named: std::collections::BTreeMap<_, _> = baseline
        .unet
        .comparison_named_parameters()
        .into_iter()
        .collect();
    assert_eq!(
        old_named
            .values()
            .map(|p| p.node_id())
            .collect::<std::collections::HashSet<_>>(),
        baseline
            .unet
            .params()
            .iter()
            .map(|p| p.node_id())
            .collect::<std::collections::HashSet<_>>()
    );
    for (index, (name, p)) in named.iter().enumerate() {
        let q = old_named[name];
        let shape = p.tensor().shape()?;
        assert_eq!(shape, q.tensor().shape(), "parameter {index}");
        let values = (0..p.tensor().numel()?)
            .map(|j| {
                if shape.len() == 1 {
                    0.1 + (j % 3) as f32 * 0.01
                } else {
                    ((index * 13 + j * 7) % 19) as f32 * 0.003 - 0.027
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
    let mut fixture = replay::Fixture {
        version: 2,
        corrections: serde_json::from_str(include_str!("../legacy/CORRECTIONS.json"))?,
        initial: old_descriptors
            .into_iter()
            .map(|descriptor| replay::Weight {
                values: old_named[&descriptor.name].tensor().data().to_vec(),
                descriptor,
            })
            .collect(),
        steps: Vec::new(),
    };
    drop(old_named);
    let image = ctx.input(vec![0.5; 128], &[2, 1, 8, 8])?;
    let old_image = old::nn::Variable::new(old::tensor::Tensor::from_vec(
        vec![0.5; 128],
        &[2, 1, 8, 8],
    )?);
    let mut optimizer = Adam::new(&ctx, 1e-3, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&parameters)?;
    let mut old_optimizer = old::optimizer::Adam::new(1e-3, 0.9, 0.999, 1e-8);
    for parameter in baseline.unet.params() {
        old_optimizer.register(parameter);
    }
    for step in 0..3 {
        let (expected, old_loss) = baseline.forward_loss_diffusion(&old_image)?;
        let (t, noise) = old::comparison::diffusion_draw(&baseline, &old_image, &old_loss)?;
        // Preserve the exact original draw even when the following parity check
        // fails. These draw-only files are diagnostic records; the CLI consumes
        // the complete fixture.json written after all three steps below.
        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/p1/reference-draws");
        std::fs::create_dir_all(&directory)?;
        std::fs::write(
            directory.join(format!("step-{step}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "source":"unmodified ReferenceDiffusion::forward_loss_diffusion",
                "shape":noise.shape(),"timestep":t,"noise":noise.data(),
                "initialization":"parameter i in sorted structural-name order,j: rank1 0.1+(j%3)*0.01; otherwise ((i*13+j*7)%19)*0.003-0.027",
                "step":step,"prior_steps_require_parity":true
            }))?,
        )?;
        let recorded_noise = noise.data().to_vec();
        let noise = ctx.tensor(noise.data().to_vec(), noise.shape())?;
        let (actual, loss) = model.forward_loss_with_noise(&image, &noise, t)?;
        close(
            &actual.tensor().to_vec()?,
            expected.tensor().data(),
            &format!("step {step} prediction"),
        );
        close(&loss.tensor().to_vec()?, old_loss.tensor().data(), "loss");
        loss.backward()?;
        old_loss.backward()?;
        let gradients = baseline
            .unet
            .comparison_named_parameters()
            .into_iter()
            .map(|(name, p)| (name, p.grad().data().to_vec()))
            .collect();
        for (name, q) in baseline.unet.comparison_named_parameters() {
            let p = named[&name];
            close(
                p.grad()?.ok_or("missing gradient")?.data(),
                q.grad().data(),
                &format!("step {step} gradient {name}"),
            );
        }
        optimizer.step()?;
        optimizer.zero_grad()?;
        old_optimizer.step()?;
        old_optimizer.zero_grad()?;
        old::comparison::clear_graph();
        fixture.steps.push(replay::Step {
            timestep: t,
            noise: recorded_noise,
            prediction: expected.tensor().data().to_vec(),
            loss: old_loss.tensor().data().to_vec(),
            gradients,
            updated: baseline
                .unet
                .comparison_named_parameters()
                .into_iter()
                .map(|(name, p)| (name, p.tensor().data().to_vec()))
                .collect(),
        });
        for (name, q) in baseline.unet.comparison_named_parameters() {
            let p = named[&name];
            close(
                &p.tensor().to_vec()?,
                q.tensor().data(),
                &format!("step {step} updated {name}"),
            );
        }
    }
    assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/p1/reference-draws/fixture.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&fixture)?)?;
    let mut loaded: replay::Fixture = serde_json::from_reader(std::fs::File::open(&path)?)?;
    loaded.initial.reverse(); // File enumeration order must not affect correspondence.
    replay::replay(&loaded)?;
    loaded.steps[2].timestep = 10;
    assert!(
        replay::replay(&loaded)
            .unwrap_err()
            .to_string()
            .contains("step 2: timestep")
    );
    loaded.steps[2].timestep = fixture.steps[2].timestep;
    loaded.steps[2].noise[0] = f32::NAN;
    assert!(
        replay::replay(&loaded)
            .unwrap_err()
            .to_string()
            .contains("step 2: nonfinite")
    );
    loaded.steps[2].noise[0] = fixture.steps[2].noise[0];
    let first_name = loaded.steps[2].gradients.keys().next().unwrap().clone();
    loaded.steps[2]
        .gradients
        .get_mut(&first_name)
        .unwrap()
        .pop();
    assert!(
        replay::replay(&loaded)
            .unwrap_err()
            .to_string()
            .contains(&format!("step 2: parameter {first_name} length"))
    );
    loaded.steps[2].gradients.insert(
        first_name.clone(),
        fixture.steps[2].gradients[&first_name].clone(),
    );
    loaded.steps[0].prediction[0] += 1.0;
    assert!(
        replay::replay(&loaded)
            .unwrap_err()
            .to_string()
            .contains("step 0 prediction")
    );
    Ok(())
}
