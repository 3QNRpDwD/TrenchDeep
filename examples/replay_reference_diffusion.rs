//! Replay a captured three-step baseline without linking or executing legacy code.
use serde::{Deserialize, Serialize};
#[path = "support/reference_parameters.rs"]
pub mod mapping;
use trench_deep::{
    nn::{Diffusion, DiffusionScheduler, Unet},
    optimizer::{Adam, Optimizer},
    trainer::TrainableModel,
    *,
};

#[derive(Serialize, Deserialize)]
pub struct Weight {
    #[serde(flatten)]
    pub descriptor: mapping::Descriptor,
    pub values: Vec<f32>,
}
#[derive(Serialize, Deserialize)]
pub struct Step {
    pub timestep: usize,
    pub noise: Vec<f32>,
    pub prediction: Vec<f32>,
    pub loss: Vec<f32>,
    pub gradients: std::collections::BTreeMap<String, Vec<f32>>,
    pub updated: std::collections::BTreeMap<String, Vec<f32>>,
}
#[derive(Serialize, Deserialize)]
pub struct Fixture {
    pub version: u32,
    pub corrections: serde_json::Value,
    pub initial: Vec<Weight>,
    pub steps: Vec<Step>,
}

fn close(a: &[f32], b: &[f32], label: &str) -> Result<(), Box<dyn std::error::Error>> {
    if a.len() != b.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        if !a.is_finite()
            || !b.is_finite()
            || (a - b).abs() > 1e-3f32.max(1e-3 * a.abs().max(b.abs()))
        {
            return Err(format!("{label}[{i}]: {a} != {b}").into());
        }
    }
    Ok(())
}

pub fn replay(fixture: &Fixture) -> Result<(), Box<dyn std::error::Error>> {
    if fixture.version != 2 || fixture.steps.len() != 3 {
        return Err("expected version 2 named reference DDPM fixture with exactly three steps; regenerate older fixtures".into());
    }
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../legacy/CORRECTIONS.json"))?;
    if fixture.corrections != expected {
        return Err("baseline correction provenance mismatch".into());
    }
    // Reject malformed later steps before performing any training work.
    for (i, step) in fixture.steps.iter().enumerate() {
        if step.timestep >= 10 {
            return Err(format!("step {i}: timestep must be less than 10").into());
        }
        if step.noise.len() != 128 || step.prediction.len() != 128 || step.loss.len() != 1 {
            return Err(format!("step {i}: reference tensor length mismatch").into());
        }
        if step.gradients.len() != fixture.initial.len()
            || step.updated.len() != fixture.initial.len()
        {
            return Err(format!("step {i}: parameter count mismatch").into());
        }
        for weight in &fixture.initial {
            let name = &weight.descriptor.name;
            if step
                .gradients
                .get(name)
                .is_none_or(|v| v.len() != weight.values.len())
                || step
                    .updated
                    .get(name)
                    .is_none_or(|v| v.len() != weight.values.len())
            {
                return Err(format!("step {i}: parameter {name} length/name mismatch").into());
            }
        }
        if step
            .noise
            .iter()
            .chain(&step.prediction)
            .chain(&step.loss)
            .chain(step.gradients.values().flatten())
            .chain(step.updated.values().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(format!("step {i}: nonfinite fixture value").into());
        }
    }
    let ctx = ExecutionContext::new();
    let model = Diffusion::new(
        &ctx,
        Unet::new(&ctx, 1, 8, &[1, 2], 4, &[])?,
        DiffusionScheduler::linear(10, 1e-4, 0.02)?,
        0,
    )?;
    let parameters = model.parameters();
    let named_entries = model.unet.named_parameters();
    let descriptors = mapping::describe(
        named_entries
            .iter()
            .map(|(name, p)| Ok((name.clone(), p.tensor().shape()?, p.id())))
            .collect::<MlResult<Vec<_>>>()?,
    )?;
    let named: std::collections::BTreeMap<_, _> = named_entries.into_iter().collect();
    mapping::validate(
        &descriptors,
        &fixture
            .initial
            .iter()
            .map(|w| w.descriptor.clone())
            .collect::<Vec<_>>(),
    )?;
    for weight in &fixture.initial {
        let p = named[&weight.descriptor.name];
        let canonical = fixture
            .initial
            .iter()
            .find(|w| w.descriptor.name == weight.descriptor.shared_with)
            .ok_or("missing shared weight")?;
        if weight.values != canonical.values {
            return Err("shared initial values mismatch".into());
        }
        if weight.values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite initial parameter".into());
        }
        ctx.replace_parameter(
            p.variable(),
            TensorBuffer::from_vec(weight.values.clone(), &weight.descriptor.shape)?,
        )?;
    }
    let image = ctx.input(vec![0.5; 128], &[2, 1, 8, 8])?;
    let mut optimizer = Adam::new(&ctx, 1e-3, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&parameters)?;
    for (i, step) in fixture.steps.iter().enumerate() {
        let noise = ctx.tensor(step.noise.clone(), &[2, 1, 8, 8])?;
        let (prediction, loss) = model.forward_loss_with_noise(&image, &noise, step.timestep)?;
        close(
            &prediction.tensor().to_vec()?,
            &step.prediction,
            &format!("step {i} prediction"),
        )?;
        close(
            &loss.tensor().to_vec()?,
            &step.loss,
            &format!("step {i} loss"),
        )?;
        loss.backward()?;
        for (name, p) in &named {
            close(
                p.grad()?.ok_or("missing gradient")?.data(),
                &step.gradients[name],
                &format!("step {i} gradient {name}"),
            )?;
        }
        optimizer.step()?;
        optimizer.zero_grad()?;
        for (name, p) in &named {
            close(
                &p.tensor().to_vec()?,
                &step.updated[name],
                &format!("step {i} updated {name}"),
            )?;
        }
    }
    if ctx.graph_stats()?.graph_nodes != 0 {
        return Err("replay left a live graph".into());
    }
    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: replay_reference_diffusion <fixture.json>")?;
    if args.next().is_some() {
        return Err(
            "usage: replay_reference_diffusion <fixture.json> (exactly one path required)".into(),
        );
    }
    let fixture: Fixture = serde_json::from_reader(std::fs::File::open(path)?)?;
    replay(&fixture)?;
    println!(
        "Reference DDPM replay passed: predictions, losses, all gradients and Adam weights over 3 steps."
    );
    Ok(())
}
