use crate::nn::{Activation, ActivationKind, Layer, Linear, Sequential};
use crate::optimizer::{Adam, Optimizer};
use crate::{ExecutionContext, MlResult, Reduction};
use std::time::Instant;

pub struct MLP {
    context: ExecutionContext,
    network: Sequential,
}

impl MLP {
    pub fn new(ctx: &ExecutionContext, input: usize, hidden: usize, output: usize) -> MlResult<Self> {
        let mut mlp = Sequential::new(&ctx, "MLP");
        mlp.push(Box::new(Linear::new(&ctx, input, hidden, "l1")?))?;
        mlp.push(Box::new(Activation::new(&ctx, ActivationKind::ReLU, "a1")))?;
        mlp.push(Box::new(Linear::new(&ctx, hidden, hidden, "l2")?))?;
        mlp.push(Box::new(Activation::new(&ctx, ActivationKind::ReLU, "a2")))?;
        mlp.push(Box::new(Linear::new(&ctx, hidden, output, "l3")?))?;
        mlp.push(Box::new(Activation::new(&ctx, ActivationKind::ReLU, "a3")))?;

        Ok(Self {
            context: ctx.clone(),
            network: mlp,
        })
    }
}

#[test]
#[cfg(all(
    feature = "builtinKernels",
    feature = "builtinStorage",
    feature = "enableBackward",
))]
pub fn three_layer_model() -> MlResult<()> {
    let ctx = ExecutionContext::new();

    let lr = 0.01;
    let data = ctx.input(
        (0..10000).map(|_| rand::random::<f32>()).collect(),
        &[100, 100],
    )?;
    let target = ctx.tensor(
        (0..10000).map(|_| rand::random::<f32>()).collect(),
        &[100, 100],
    )?;
    let mlp = MLP::new(&ctx, 100, 100, 100)?;

    let mut optimizer = Adam::new(&ctx, lr, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&mlp.network.parameters())?;
    println!("train start");
    let start = Instant::now();

    let y = mlp.network.apply(&data)?;
    let loss = y.mse_loss(&target, Reduction::Mean)?;
    loss.backward()?;
    optimizer.step()?;

    let end = Instant::now();

    println!("{:?}", end - start);
    println!("train end");

    Ok(())
}
impl crate::trainer::TrainableModel for MLP {
    fn context_id(&self) -> crate::ContextId {
        self.context.id()
    }

    fn parameters(&self) -> Vec<&crate::Parameter> {
        self.network.parameters()
    }
}



#[test]
#[cfg(all(
    feature = "builtinKernels",
    feature = "builtinStorage",
    feature = "enableBackward",
))]
pub fn three_layer_model_prepare() -> MlResult<()> {
    use crate::trainer::PreparedObjective;
    use crate::trainer::{SupervisedBatch, TrainableModel};

    let ctx = ExecutionContext::new();
    let lr = 0.01;
    let data = ctx.input(
        (0..10000).map(|_| rand::random::<f32>()).collect(),
        &[100, 100],
    )?;
    let target = ctx.tensor(
        (0..10000).map(|_| rand::random::<f32>()).collect(),
        &[100, 100],
    )?;
    let mut mlp = MLP::new(&ctx, 100, 100, 100)?;

    let mut optimizer = Adam::new(&ctx, lr, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&mlp.parameters())?;
    let batch = objective().execution_batch(&mut mlp, 
        &SupervisedBatch {
            inputs: data,
            targets: target,
        },
        &crate::trainer::TrainingStepContext::default(),
    )?;

    let prepare_start = Instant::now();
    let mut prepared = ctx.prepare_objective(&mut mlp, &objective(), &batch.inputs)?;
    println!("prepare: {:?}", prepare_start.elapsed());

    println!("train start");
    let start = Instant::now();
    prepared.run(&mlp, &batch.inputs, |output| {
        assert!(output.loss.tensor().item()?.is_finite());
        output.loss.backward()?;
        optimizer.step()
    })?;
    println!("train: {:?}", start.elapsed());
    println!("train end");

    Ok(())
}



impl crate::trainer::ForwardModel for MLP {
    fn forward(&self, input: &crate::Variable) -> MlResult<crate::Variable> {
        
        self.network.apply(input)
    }
}

fn objective() -> crate::trainer::Supervised<crate::loss::MseLoss> {
    crate::trainer::Supervised::new(crate::loss::MseLoss::new(crate::Reduction::Mean))
}
