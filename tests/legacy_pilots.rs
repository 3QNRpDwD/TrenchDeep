#![cfg(all(
    feature = "legacyBenchmark",
    feature = "builtinStorage",
    feature = "builtinKernels"
))]
use old::{
    nn::Parameter as OldParameter,
    optimizer::Optimizer as OldOptimizer,
    tensor::{AutogradFunction, TensorBase, operators::Function},
    trainer::TrainableModel as OldModel,
};
use trench_deep::legacy as old;
use trench_deep::{
    nn::{BigramLm, LinearPolicy, PiClassifier},
    optimizer::{Optimizer, SGD},
    trainer::*,
    *,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn pi_model_matches_the_original_nonzero_sampled_augmentations() -> Result<()> {
    let ctx = ExecutionContext::new();
    let model = PiClassifier::new(&ctx, 2, 2, 0.2)?;
    let mut baseline = old::comparison::semi_supervised::PiToyClassifier::new(2, 2, 0.2)?;
    initialize(&ctx, model.parameters(), baseline.params())?;
    let (x, ox) = input(&ctx, vec![0.2, 0.8], &[1, 2])?;
    let (t, ot) = input(&ctx, vec![0.0, 1.0], &[1, 2])?;
    let (_, ou) = input(&ctx, vec![0.3, 0.7], &[1, 2])?;
    let (oy, oloss) =
        old::trainer::SemiSupervisedModel::forward_loss(&mut baseline, &ox, &ot, &ou, 0.4)?;
    let views = old::comparison::direct_add_outputs(&ou)?;
    assert_eq!(
        views.len(),
        2,
        "expected original Pi-model's two noisy Add outputs"
    );
    let first = ctx.input(views[0].data().to_vec(), views[0].shape())?;
    let second = ctx.input(views[1].data().to_vec(), views[1].shape())?;
    let (y, loss) = model.forward_loss_with_augmentations(&x, t.tensor(), &first, &second, 0.4)?;
    close(&y.tensor().to_vec()?, oy.tensor().data());
    verify(&ctx, &loss, &oloss, model.parameters(), baseline.params())
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= 1e-3f32.max(1e-3 * a.abs().max(b.abs())),
            "{a} != {b}"
        );
    }
}
fn initialize(ctx: &ExecutionContext, a: Vec<&Parameter>, b: Vec<&dyn OldParameter>) -> Result<()> {
    assert_eq!(a.len(), b.len());
    for (i, (p, q)) in a.iter().zip(b).enumerate() {
        let shape = p.tensor().shape()?;
        assert_eq!(shape, q.tensor().shape());
        let data = (0..p.tensor().numel()?)
            .map(|j| ((i * 7 + j * 3) % 13) as f32 * 0.02 - 0.1)
            .collect::<Vec<_>>();
        ctx.replace_parameter(p.variable(), TensorBuffer::from_vec(data.clone(), &shape)?)?;
        q.tensor()
            .replace(old::tensor::GlobalTensor::from_vec(data, &shape)?);
    }
    Ok(())
}
fn verify(
    ctx: &ExecutionContext,
    a: &Variable,
    b: &old::nn::Variable,
    p: Vec<&Parameter>,
    q: Vec<&dyn OldParameter>,
) -> Result<()> {
    close(&a.tensor().to_vec()?, b.tensor().data());
    a.backward()?;
    b.backward()?;
    for (p, q) in p.iter().zip(&q) {
        close(p.grad()?.ok_or("missing gradient")?.data(), q.grad().data());
    }
    let mut optimizer = SGD::new(ctx, 0.01)?;
    optimizer.register_all(&p)?;
    let mut old_optimizer = old::optimizer::SGD::new(0.01);
    for &q in &q {
        old_optimizer.register(q);
    }
    optimizer.step()?;
    old_optimizer.step()?;
    for (p, q) in p.iter().zip(q) {
        close(&p.tensor().to_vec()?, q.tensor().data());
    }
    old::comparison::clear_graph();
    Ok(())
}
fn input(
    ctx: &ExecutionContext,
    values: Vec<f32>,
    shape: &[usize],
) -> Result<(Variable, old::nn::Variable)> {
    Ok((
        ctx.input(values.clone(), shape)?,
        old::nn::Variable::new(old::tensor::Tensor::from_vec(values, shape)?),
    ))
}
#[test]
fn legacy_sigmoid_discrepancy_blocks_mlp_performance_comparison() -> Result<()> {
    let ctx = ExecutionContext::new();
    let input = ctx.scalar(1.0)?;
    let actual = input.sigmoid()?.item()?;
    let old_input = old::tensor::Tensor::from_vec(vec![1.0], &[])?;
    let old_output = old::nn::activation::SigmoidOp::new()?.forward(&[&old_input])?;
    let expected = old_output.first().ok_or("missing sigmoid output")?.data()[0];
    assert!((actual - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    assert!((expected - 1.0 / (1.0 + 1.0f32.exp())).abs() < 1e-6);
    assert!(
        (actual - expected).abs() > 0.4,
        "legacy mismatch must remain visible until the user changes the baseline"
    );
    Ok(())
}
#[test]
fn bigram_matches_shift_loss_gradients_and_update() -> Result<()> {
    let ctx = ExecutionContext::new();
    let mut model = BigramLm::new(&ctx, 3)?;
    let mut baseline = old::comparison::autoregressive::BigramLM::new(3)?;
    initialize(&ctx, model.parameters(), baseline.params())?;
    let (x, ox) = input(
        &ctx,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
    )?;
    let (y, loss, tokens) = model.forward_loss(&x)?;
    let (oy, oloss, old_tokens) =
        old::trainer::AutoregressiveModel::forward_loss(&mut baseline, &ox)?;
    assert_eq!(tokens, old_tokens);
    let values = y.tensor().to_vec()?;
    close(&values[values.len() - 3..], oy.tensor().data());
    verify(&ctx, &loss, &oloss, model.parameters(), baseline.params())
}
#[test]
fn pi_model_zero_noise_fixture_matches_loss_gradients_and_update() -> Result<()> {
    let ctx = ExecutionContext::new();
    let mut model = PiClassifier::new(&ctx, 2, 2, 0.0)?;
    let mut baseline = old::comparison::semi_supervised::PiToyClassifier::new(2, 2, 0.0)?;
    initialize(&ctx, model.parameters(), baseline.params())?;
    let (x, ox) = input(&ctx, vec![0.2, 0.8], &[1, 2])?;
    let (t, ot) = input(&ctx, vec![0.0, 1.0], &[1, 2])?;
    let (y, loss) = model.forward_loss(&x, t.tensor(), &x, 0.4)?;
    let (oy, oloss) =
        old::trainer::SemiSupervisedModel::forward_loss(&mut baseline, &ox, &ot, &ox, 0.4)?;
    close(&y.tensor().to_vec()?, oy.tensor().data());
    verify(&ctx, &loss, &oloss, model.parameters(), baseline.params())
}
#[test]
fn policy_fixed_action_loss_gradients_and_update_match() -> Result<()> {
    let ctx = ExecutionContext::new();
    let mut model = LinearPolicy::new(&ctx, 1, 2)?;
    let mut baseline = old::comparison::reinforcement::LinearPolicy::new(1, 2)?;
    initialize(&ctx, model.parameters(), baseline.params())?;
    let (x, ox) = input(&ctx, vec![1.0], &[1, 1])?;
    let y = model.policy_logits(&x)?;
    let oy = old::trainer::RLModel::policy_logits(&mut baseline, &ox)?;
    close(&y.tensor().to_vec()?, oy.tensor().data());
    let (t, ot) = input(&ctx, vec![0.0, 1.0], &[1, 2])?;
    let loss = y.softmax_cross_entropy(t.tensor(), Reduction::Mean)?;
    let oloss = old::loss::SoftmaxCrossEntropyLoss::new()?.apply(&[&oy, &ot])?;
    verify(&ctx, &loss, &oloss, model.parameters(), baseline.params())
}

#[derive(Default)]
struct Rollout {
    actions: Vec<usize>,
}
impl Environment for Rollout {
    fn reset(&mut self) -> MlResult<TensorBuffer> {
        self.actions.clear();
        TensorBuffer::from_vec(vec![1.0], &[1, 1])
    }
    fn step(&mut self, action: usize) -> MlResult<StepResult> {
        self.actions.push(action);
        Ok(StepResult {
            next_observation: TensorBuffer::from_vec(vec![1.0], &[1, 1])?,
            reward: if action == 0 { 0.2 } else { 0.8 },
            done: self.actions.len() == 3,
        })
    }
    fn num_actions(&self) -> usize {
        2
    }
    fn observation_shape(&self) -> Vec<usize> {
        vec![1, 1]
    }
}
impl old::trainer::Environment for Rollout {
    fn reset(&mut self) -> old::MlResult<old::tensor::Tensor> {
        self.actions.clear();
        old::tensor::Tensor::from_vec(vec![1.0], &[1, 1])
    }
    fn step(&mut self, action: usize) -> old::MlResult<old::trainer::StepResult> {
        self.actions.push(action);
        Ok(old::trainer::StepResult {
            next_observation: old::tensor::Tensor::from_vec(vec![1.0], &[1, 1])?,
            reward: if action == 0 { 0.2 } else { 0.8 },
            done: self.actions.len() == 3,
        })
    }
    fn num_actions(&self) -> usize {
        2
    }
    fn observation_shape(&self) -> Vec<usize> {
        vec![1, 1]
    }
}
#[test]
fn rl_rollout_and_trainer_updates_match_with_identical_action_draws() -> Result<()> {
    let ctx = ExecutionContext::new();
    let mut model = LinearPolicy::new(&ctx, 1, 2)?;
    let mut baseline = old::comparison::reinforcement::LinearPolicy::new(1, 2)?;
    initialize(&ctx, model.parameters(), baseline.params())?;
    let mut optimizer = SGD::new(&ctx, 0.01)?;
    optimizer.register_all(&model.parameters())?;
    let mut old_optimizer = old::optimizer::SGD::new(0.01);
    for p in baseline.params() {
        old_optimizer.register(p);
    }
    let mut env = Rollout::default();
    let mut old_env = Rollout::default();
    // Align the new action substream with the baseline's single RNG for this episode.
    let trainer = RLTrainer::silent(&ctx).with_seed(42 ^ 0xA076_1D64_78BD_642F);
    let old_trainer = old::trainer::Trainer::builder()
        .seed(42)
        .show_progress(false)
        .build()
        .reinforcement();
    let actual = trainer.fit(
        &mut model,
        &mut env,
        &mut optimizer,
        EpisodeSchedule::new(1, 3)?,
    )?;
    let expected = old_trainer.fit(
        &mut baseline,
        &mut old_env,
        &mut old_optimizer,
        old::trainer::EpisodeSchedule::new(1, 3)?,
    )?;
    assert_eq!(env.actions, old_env.actions);
    close(&[actual.final_loss], &[expected.final_loss]);
    for (p, q) in model.parameters().iter().zip(baseline.params()) {
        close(&p.tensor().to_vec()?, q.tensor().data());
    }
    Ok(())
}
