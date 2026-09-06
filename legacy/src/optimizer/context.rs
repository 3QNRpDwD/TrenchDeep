use crate::nn::ContextParameter;
use crate::tensor::{GlobalTensor, TensorBase};
use crate::{ContextError, ContextId, ExecutionContext, MlResult};

use super::OptimError;

pub trait ContextOptimizer {
    fn register(&mut self, parameter: &ContextParameter) -> MlResult<()>;
    fn register_all(&mut self, parameters: &[&ContextParameter]) -> MlResult<()> {
        for parameter in parameters {
            self.register(parameter)?;
        }
        Ok(())
    }
    fn step(&mut self) -> MlResult<()>;
    fn zero_grad(&self) -> MlResult<()>;
    fn lr(&self) -> f32;
    fn set_lr(&mut self, learning_rate: f32) -> MlResult<()>;
    fn registered_param_count(&self) -> usize;
    fn registered_parameters(&self) -> Vec<&ContextParameter>;
    fn context_id(&self) -> ContextId;
}

#[derive(Clone, Copy, Debug)]
enum Algorithm {
    Sgd,
    Momentum { momentum: f32 },
    AdaGrad { epsilon: f32 },
    RmsProp { rho: f32, epsilon: f32 },
    Adam { beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
}

#[derive(Debug)]
struct ParameterState {
    parameter: ContextParameter,
    first: Vec<f32>,
    second: Vec<f32>,
}

#[derive(Debug)]
struct OptimizerCore {
    context: ExecutionContext,
    learning_rate: f32,
    algorithm: Algorithm,
    step: u32,
    parameters: Vec<ParameterState>,
}

fn positive(name: &'static str, value: f32) -> MlResult<f32> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(OptimError::InvalidHyperparameter { name, reason: "must be finite and greater than zero".into() }.into())
    }
}

fn unit_interval(name: &'static str, value: f32) -> MlResult<f32> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err(OptimError::InvalidHyperparameter { name, reason: "must be finite and in [0, 1)".into() }.into())
    }
}

impl OptimizerCore {
    fn new(context: &ExecutionContext, learning_rate: f32, algorithm: Algorithm) -> MlResult<Self> {
        Ok(Self {
            context: context.clone(),
            learning_rate: positive("learning_rate", learning_rate)?,
            algorithm,
            step: 0,
            parameters: Vec::new(),
        })
    }

    fn register(&mut self, parameter: &ContextParameter) -> MlResult<()> {
        if parameter.context_id() != self.context.id() {
            return Err(ContextError::Mismatch.into());
        }
        if self.parameters.iter().any(|entry| entry.parameter.node_id() == parameter.node_id()) {
            return Err(OptimError::DuplicateParameter(parameter.node_id()).into());
        }
        let size = parameter.tensor().to_vec()?.len();
        self.parameters.push(ParameterState {
            parameter: parameter.clone(),
            first: vec![0.0; size],
            second: vec![0.0; size],
        });
        Ok(())
    }

    fn gradients(&self) -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
        self.parameters.iter().map(|entry| {
            if entry.parameter.context_id() != self.context.id() {
                return Err(ContextError::Mismatch.into());
            }
            let shape = entry.parameter.tensor().shape()?;
            let gradient = self.context.grad(entry.parameter.variable())?;
            if let Some(ref gradient) = gradient {
                if gradient.shape != shape {
                    return Err(OptimError::GradientError(format!(
                        "parameter {:?} expected gradient shape {:?}, got {:?}",
                        entry.parameter.node_id(), shape, gradient.shape
                    )).into());
                }
            }
            Ok(gradient)
        }).collect()
    }

    fn step(&mut self) -> MlResult<()> {
        let gradients = self.gradients()?;
        if matches!(self.algorithm, Algorithm::Adam { .. }) {
            self.step = self.step.checked_add(1).ok_or_else(|| {
                OptimError::GradientError("optimizer step counter overflow".into())
            })?;
        }
        for (entry, gradient) in self.parameters.iter_mut().zip(gradients) {
            let Some(gradient) = gradient else { continue };
            let mut delta = vec![0.0; gradient.data.len()];
            match self.algorithm {
                Algorithm::Sgd => {
                    for (output, &g) in delta.iter_mut().zip(&gradient.data) {
                        *output = self.learning_rate * g;
                    }
                }
                Algorithm::Momentum { momentum } => {
                    for ((velocity, output), &g) in entry.first.iter_mut().zip(&mut delta).zip(&gradient.data) {
                        *velocity = momentum * *velocity + g;
                        *output = self.learning_rate * *velocity;
                    }
                }
                Algorithm::AdaGrad { epsilon } => {
                    for ((accumulator, output), &g) in entry.first.iter_mut().zip(&mut delta).zip(&gradient.data) {
                        *accumulator += g * g;
                        *output = self.learning_rate * g / (*accumulator + epsilon).sqrt();
                    }
                }
                Algorithm::RmsProp { rho, epsilon } => {
                    for ((average, output), &g) in entry.first.iter_mut().zip(&mut delta).zip(&gradient.data) {
                        *average = rho * *average + (1.0 - rho) * g * g;
                        *output = self.learning_rate * g / (*average + epsilon).sqrt();
                    }
                }
                Algorithm::Adam { beta1, beta2, epsilon, weight_decay } => {
                    let correction1 = 1.0 - beta1.powi(self.step as i32);
                    let correction2 = 1.0 - beta2.powi(self.step as i32);
                    let weights = if weight_decay == 0.0 { None } else { Some(entry.parameter.tensor().to_vec()?) };
                    for index in 0..gradient.data.len() {
                        let g = gradient.data[index];
                        entry.first[index] = beta1 * entry.first[index] + (1.0 - beta1) * g;
                        entry.second[index] = beta2 * entry.second[index] + (1.0 - beta2) * g * g;
                        let adaptive = (entry.first[index] / correction1)
                            / ((entry.second[index] / correction2).sqrt() + epsilon);
                        delta[index] = self.learning_rate * (adaptive
                            + weights.as_ref().map_or(0.0, |values| weight_decay * values[index]));
                    }
                }
            }
            self.context.sub_assign(
                entry.parameter.variable(),
                &GlobalTensor::from_vec(delta, &gradient.shape)?,
            )?;
        }
        Ok(())
    }

    fn zero_grad(&self) -> MlResult<()> {
        for entry in &self.parameters {
            self.context.clear_grad(entry.parameter.variable())?;
        }
        Ok(())
    }
}

macro_rules! context_optimizer {
    ($name:ident) => {
        #[derive(Debug)]
        pub struct $name(OptimizerCore);
        impl ContextOptimizer for $name {
            fn register(&mut self, parameter: &ContextParameter) -> MlResult<()> { self.0.register(parameter) }
            fn step(&mut self) -> MlResult<()> { self.0.step() }
            fn zero_grad(&self) -> MlResult<()> { self.0.zero_grad() }
            fn lr(&self) -> f32 { self.0.learning_rate }
            fn set_lr(&mut self, learning_rate: f32) -> MlResult<()> {
                self.0.learning_rate = positive("learning_rate", learning_rate)?;
                Ok(())
            }
            fn registered_param_count(&self) -> usize { self.0.parameters.len() }
            fn registered_parameters(&self) -> Vec<&ContextParameter> {
                self.0.parameters.iter().map(|entry| &entry.parameter).collect()
            }
            fn context_id(&self) -> ContextId { self.0.context.id() }
        }
    };
}

context_optimizer!(ContextSGD);
context_optimizer!(ContextMomentum);
context_optimizer!(ContextAdaGrad);
context_optimizer!(ContextRMSProp);
context_optimizer!(ContextAdam);
context_optimizer!(ContextAdamW);

impl ContextSGD {
    pub fn new(context: &ExecutionContext, learning_rate: f32) -> MlResult<Self> {
        Ok(Self(OptimizerCore::new(context, learning_rate, Algorithm::Sgd)?))
    }
}
impl ContextMomentum {
    pub fn new(context: &ExecutionContext, learning_rate: f32, momentum: f32) -> MlResult<Self> {
        Ok(Self(OptimizerCore::new(context, learning_rate, Algorithm::Momentum { momentum: unit_interval("momentum", momentum)? })?))
    }
}
impl ContextAdaGrad {
    pub fn new(context: &ExecutionContext, learning_rate: f32, epsilon: f32) -> MlResult<Self> {
        Ok(Self(OptimizerCore::new(context, learning_rate, Algorithm::AdaGrad { epsilon: positive("epsilon", epsilon)? })?))
    }
}
impl ContextRMSProp {
    pub fn new(context: &ExecutionContext, learning_rate: f32, rho: f32, epsilon: f32) -> MlResult<Self> {
        Ok(Self(OptimizerCore::new(context, learning_rate, Algorithm::RmsProp {
            rho: unit_interval("rho", rho)?, epsilon: positive("epsilon", epsilon)?,
        })?))
    }
}
impl ContextAdam {
    pub fn new(context: &ExecutionContext, learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> MlResult<Self> {
        Ok(Self(OptimizerCore::new(context, learning_rate, Algorithm::Adam {
            beta1: unit_interval("beta1", beta1)?, beta2: unit_interval("beta2", beta2)?,
            epsilon: positive("epsilon", epsilon)?, weight_decay: 0.0,
        })?))
    }
}
impl ContextAdamW {
    pub fn new(context: &ExecutionContext, learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) -> MlResult<Self> {
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(OptimError::InvalidHyperparameter { name: "weight_decay", reason: "must be finite and non-negative".into() }.into());
        }
        Ok(Self(OptimizerCore::new(context, learning_rate, Algorithm::Adam {
            beta1: unit_interval("beta1", beta1)?, beta2: unit_interval("beta2", beta2)?,
            epsilon: positive("epsilon", epsilon)?, weight_decay,
        })?))
    }
}

pub fn clip_context_grad_norm(
    context: &ExecutionContext,
    parameters: &[&ContextParameter],
    max_norm: f32,
) -> MlResult<f32> {
    positive("max_norm", max_norm)?;
    let mut squared_norm = 0.0;
    for parameter in parameters {
        if parameter.context_id() != context.id() { return Err(ContextError::Mismatch.into()); }
        if let Some(gradient) = context.grad(parameter.variable())? {
            squared_norm += gradient.data.iter().map(|value| value * value).sum::<f32>();
        }
    }
    let norm = squared_norm.sqrt();
    if norm > max_norm {
        let factor = max_norm / (norm + 1e-6);
        for parameter in parameters {
            context.scale_grad(parameter.variable(), factor)?;
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameter_with_gradient(context: &ExecutionContext) -> MlResult<ContextParameter> {
        let parameter = ContextParameter::new(context.parameter(vec![1.0, -2.0], &[2])?);
        let squared = context.square_variable(parameter.variable())?;
        context.sum_variable(&squared)?.backward()?;
        Ok(parameter)
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn all_context_optimizers_match_their_first_step() -> MlResult<()> {
        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        optimizer.register(&parameter)?;
        optimizer.step()?;
        assert_close(&parameter.tensor().to_vec()?, &[0.8, -1.6]);

        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let mut optimizer = ContextMomentum::new(&context, 0.1, 0.9)?;
        optimizer.register(&parameter)?;
        optimizer.step()?;
        assert_close(&parameter.tensor().to_vec()?, &[0.8, -1.6]);

        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let mut optimizer = ContextAdaGrad::new(&context, 0.1, 1e-8)?;
        optimizer.register(&parameter)?;
        optimizer.step()?;
        assert_close(&parameter.tensor().to_vec()?, &[0.9, -1.9]);

        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let mut optimizer = ContextRMSProp::new(&context, 0.1, 0.9, 1e-8)?;
        optimizer.register(&parameter)?;
        optimizer.step()?;
        assert_close(&parameter.tensor().to_vec()?, &[0.6837722, -1.6837722]);

        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let mut optimizer = ContextAdam::new(&context, 0.1, 0.9, 0.999, 1e-8)?;
        optimizer.register(&parameter)?;
        optimizer.step()?;
        assert_close(&parameter.tensor().to_vec()?, &[0.9, -1.9]);

        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let mut optimizer = ContextAdamW::new(&context, 0.1, 0.9, 0.999, 1e-8, 0.1)?;
        optimizer.register(&parameter)?;
        optimizer.step()?;
        assert_close(&parameter.tensor().to_vec()?, &[0.89, -1.88]);
        Ok(())
    }

    #[test]
    fn context_optimizer_zero_grad_and_clipping_use_context_storage() -> MlResult<()> {
        let context = ExecutionContext::new();
        let parameter = parameter_with_gradient(&context)?;
        let norm = clip_context_grad_norm(&context, &[&parameter], 1.0)?;
        assert_close(&[norm], &[(20.0_f32).sqrt()]);
        let clipped = parameter.grad()?.expect("clipped gradient");
        let clipped_norm = clipped.data.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() <= 1e-5);

        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        optimizer.register(&parameter)?;
        optimizer.zero_grad()?;
        assert!(parameter.grad()?.is_none());
        Ok(())
    }

    #[test]
    fn registration_rejects_duplicates_and_foreign_contexts() -> MlResult<()> {
        let context = ExecutionContext::new();
        let foreign = ExecutionContext::new();
        let parameter = ContextParameter::new(context.parameter(vec![1.0], &[1])?);
        let foreign_parameter = ContextParameter::new(foreign.parameter(vec![1.0], &[1])?);
        let mut optimizer = ContextSGD::new(&context, 0.1)?;
        optimizer.register(&parameter)?;
        assert!(matches!(optimizer.register(&parameter), Err(crate::MlError::OptimError(OptimError::DuplicateParameter(_)))));
        assert!(matches!(optimizer.register(&foreign_parameter), Err(crate::MlError::ContextError(ContextError::Mismatch))));
        assert!(ContextAdam::new(&context, -0.1, 0.9, 0.999, 1e-8).is_err());
        Ok(())
    }
}
