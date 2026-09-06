//! Public operation surface; all dispatch stays in the execution runtime.
use super::*;

impl ContextTensor {
    pub(crate) fn execution_context(&self) -> MlResult<ExecutionContext> {
        let runtime = self.0.runtime.upgrade().ok_or(ContextError::Dropped)?;
        Ok(ExecutionContext {
            id: self.context_id(), runtime, _not_sync: Rc::new(Cell::new(())),
        })
    }

    /// Wrap this handle without detaching or copying its buffer.
    pub fn as_variable(&self) -> MlResult<ContextVariable> {
        self.execution_context()?.variable_from(self.clone())
    }
}

macro_rules! receiver_ops {
    ($($name:ident($($arg:ident: $ty:ty),*));* $(;)?) => {$ (
        impl ContextTensor {
            pub fn $name(&self, $($arg: $ty),*) -> MlResult<Self> {
                self.execution_context()?.$name(self, $($arg),*)
            }
        }
        impl ContextVariable {
            pub fn $name(&self, $($arg: $ty),*) -> MlResult<Self> {
                self.tensor.$name($($arg),*)?.as_variable()
            }
        }
    )*};
}

receiver_ops! {
    add(rhs: &ContextTensor);
    sub(rhs: &ContextTensor);
    mul(rhs: &ContextTensor);
    div(rhs: &ContextTensor);
    matmul(rhs: &ContextTensor);
    neg(); square(); exp(); log(); sqrt(); abs();
    sin(); cos(); tanh(); sigmoid(); silu(); relu(); sum();
    powf(exponent: f32);
    approx_sin(threshold: f32);
    approx_cos(threshold: f32);
    softmax(axis: usize);
    reshape(shape: &[usize]);
    transpose(axes: &[usize]);
    mse_loss(target: &ContextTensor, reduction: Reduction);
    mae_loss(target: &ContextTensor, reduction: Reduction);
    huber_loss(target: &ContextTensor, delta: f32, reduction: Reduction);
    binary_cross_entropy(target: &ContextTensor, reduction: Reduction);
    cross_entropy(target: &ContextTensor, reduction: Reduction);
    softmax_cross_entropy(target: &ContextTensor, reduction: Reduction);
    conv2d(weight: &ContextTensor, bias: &ContextTensor, stride: (usize, usize), padding: (usize, usize));
    max_pool2d(kernel: (usize, usize), stride: (usize, usize));
    avg_pool2d(kernel: (usize, usize), stride: (usize, usize));
    nearest_upsample2d(scale: (usize, usize));
    group_norm(gamma: &ContextTensor, beta: &ContextTensor, groups: usize, epsilon: f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_keeps_parameter_tracking_and_matches_context_dispatch() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.input(vec![2.0, 3.0], &[1, 2])?;
        let w = ctx.parameter(vec![4.0, 5.0], &[2, 1])?;
        let y = x.matmul(w.tensor())?;
        assert_eq!(y.tensor().item()?, 23.0);
        y.backward()?;
        assert_eq!(w.grad()?.ok_or(ContextError::Dropped)?.data, vec![2.0, 3.0]);
        ctx.no_grad(|| {
            let predicted = x.matmul(w.tensor())?;
            assert!(!predicted.requires_grad()?);
            assert_eq!(predicted.tensor().item()?, 23.0);
            Ok(())
        })?;
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    }

    #[test]
    fn receiver_validates_foreign_and_expired_handles() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let other = ExecutionContext::new();
        let x = ctx.scalar(1.0)?;
        let y = other.scalar(2.0)?;
        assert!(matches!(x.add(&y), Err(crate::MlError::ContextError(ContextError::Mismatch))));
        drop(other);
        assert!(matches!(y.numel(), Err(crate::MlError::ContextError(ContextError::Dropped))));
        assert!(matches!(y.square(), Err(crate::MlError::ContextError(ContextError::Dropped))));
        Ok(())
    }

    #[test]
    fn backward_preserves_gradients_of_an_independent_graph() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let left = ctx.parameter(vec![2.0], &[])?;
        let right = ctx.parameter(vec![3.0], &[])?;
        left.square()?.backward()?;
        right.square()?.backward()?;
        assert_eq!(left.grad()?.ok_or(ContextError::Dropped)?.data, vec![4.0]);
        assert_eq!(right.grad()?.ok_or(ContextError::Dropped)?.data, vec![6.0]);
        Ok(())
    }
}
