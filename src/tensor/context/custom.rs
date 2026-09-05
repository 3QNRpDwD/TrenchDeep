//! Registry-free extension interface for a single differentiable output.
use super::*;

pub trait CustomOp {
    fn name(&self) -> &'static str;
    fn input_count(&self) -> usize;
    fn forward(&self, inputs: &[TensorView<'_>]) -> MlResult<OpOutput>;
}

pub struct OpOutput {
    pub output: GlobalTensor<f32>,
    pub saved: Vec<GlobalTensor<f32>>,
    pub backward: Option<Box<dyn BackwardOp>>,
}

impl ExecutionContext {
    pub fn apply_custom(&self, op: &dyn CustomOp, inputs: &[&ContextTensor]) -> MlResult<ContextTensor> {
        if inputs.len() != op.input_count() {
            return Err(TensorError::InvalidOperation {
                op: op.name(), reason: format!("expected {} inputs, got {}", op.input_count(), inputs.len()),
            }.into());
        }
        for input in inputs {
            if input.context_id() != self.id() { return Err(ContextError::Mismatch.into()); }
            input.numel()?;
        }
        let (result, tracked) = {
            let state = self.runtime.state.try_borrow().map_err(|_| ContextError::BorrowConflict)?;
            let buffers = inputs.iter().map(|input| {
                let entry = state.tensors.get(&input.node_id())
                    .ok_or(ContextError::UnknownTensor(input.node_id()))?;
                entry.buffer.try_borrow().map_err(|_| ContextError::BorrowConflict)
            }).collect::<Result<Vec<_>, _>>()?;
            let views = buffers.iter().map(|buffer| TensorView {
                data: &buffer.data, shape: &buffer.shape,
            }).collect::<Vec<_>>();
            let tracked = state.no_grad_depth == 0
                && inputs.iter().any(|input| state.tracked.contains(&input.node_id()));
            (op.forward(&views)?, tracked)
        };
        if tracked && result.backward.is_none() {
            return Err(AutogradError::BackwardNotSupported(op.name().into()).into());
        }
        if let Some(backward) = &result.backward {
            if backward.input_count() != inputs.len() {
                return Err(AutogradError::BackwardArityMismatch {
                    expected: inputs.len(), got: backward.input_count(),
                }.into());
            }
        }
        // Validate every owned buffer before inserting any output or saved data.
        let output = GlobalTensor::from_vec(result.output.data, &result.output.shape)?;
        let saved = result.saved.into_iter().map(|buffer| {
            GlobalTensor::from_vec(buffer.data, &buffer.shape)
        }).collect::<MlResult<Vec<_>>>()?;
        let output = self.tensor(output.data, &output.shape)?;
        if tracked {
            let saved = saved.into_iter().map(|buffer| self.tensor(buffer.data, &buffer.shape))
                .collect::<MlResult<Vec<_>>>()?;
            let mut state = self.runtime.state.try_borrow_mut().map_err(|_| ContextError::BorrowConflict)?;
            let backward = result.backward.ok_or_else(|| AutogradError::BackwardNotSupported(op.name().into()))?;
            let mut pins = Vec::new();
            for id in std::iter::once(output.node_id())
                .chain(inputs.iter().map(|input| input.node_id()))
                .chain(saved.iter().map(|input| input.node_id()))
            {
                if let Err(error) = state.pin(id) {
                    for pinned in pins { state.unpin(pinned); }
                    return Err(error);
                }
                pins.push(id);
            }
            state.tracked.insert(output.node_id());
            state.graph.insert(output.node_id(), GraphNode {
                inputs: inputs.iter().map(|input| input.node_id()).collect(),
                saved: saved.iter().map(|input| input.node_id()).collect(),
                backward,
            });
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SquareOp { differentiable: bool }
    #[derive(Debug)]
    struct SquareVjp;

    impl CustomOp for SquareOp {
        fn name(&self) -> &'static str { "custom_square" }
        fn input_count(&self) -> usize { 1 }
        fn forward(&self, inputs: &[TensorView<'_>]) -> MlResult<OpOutput> {
            Ok(OpOutput {
                output: GlobalTensor::from_vec(
                    inputs[0].data().iter().map(|x| x * x).collect(), inputs[0].shape(),
                )?,
                saved: vec![GlobalTensor::from_vec(inputs[0].data().to_vec(), inputs[0].shape())?],
                backward: self.differentiable.then(|| Box::new(SquareVjp) as Box<dyn BackwardOp>),
            })
        }
    }
    impl BackwardOp for SquareVjp {
        fn name(&self) -> &'static str { "custom_square" }
        fn input_count(&self) -> usize { 1 }
        fn backward(&self, _: &[TensorView<'_>], saved: &[TensorView<'_>], grad: TensorView<'_>)
            -> MlResult<Vec<Option<GlobalTensor<f32>>>> {
            Ok(vec![Some(GlobalTensor::from_vec(
                saved[0].data().iter().zip(grad.data()).map(|(x, g)| 2.0 * x * g).collect(),
                grad.shape(),
            )?)])
        }
    }

    #[test]
    fn custom_vjp_pins_saved_values_and_frees_them_after_backward() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0, -3.0], &[2])?;
        let y = ctx.apply_custom(&SquareOp { differentiable: true }, &[x.tensor()])?;
        assert_eq!(y.to_vec()?, vec![4.0, 9.0]);
        assert_eq!(ctx.graph_stats()?.saved_tensor_references, 1);
        let loss = y.as_variable()?.sum()?;
        loss.backward()?;
        assert_eq!(x.grad()?.ok_or(ContextError::Dropped)?.data, vec![4.0, -6.0]);
        assert_eq!(ctx.graph_stats()?.saved_tensor_references, 0);
        assert_eq!(ctx.graph_stats()?.tensors, 3);
        Ok(())
    }

    #[test]
    fn forward_only_custom_op_rejects_tracking_without_leaking_storage() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let op = SquareOp { differentiable: false };
        assert!(matches!(ctx.apply_custom(&op, &[x.tensor()]),
            Err(crate::MlError::AutogradError(AutogradError::BackwardNotSupported(_)))));
        assert_eq!(ctx.graph_stats()?.tensors, 1);
        ctx.no_grad(|| {
            assert_eq!(ctx.apply_custom(&op, &[x.tensor()])?.item()?, 4.0);
            Ok(())
        })?;
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    }
}
