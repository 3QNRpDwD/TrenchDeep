use super::{invalid, plan::*};
use crate::{
    ExecutionContext, ExecutionRoute, MlResult, Parameter,
    contracts::{LossKind, Operation, Reduction},
};

pub(super) fn numel(shape: &[usize]) -> MlResult<usize> {
    if shape.contains(&0) {
        return Err(invalid("empty axes are not supported by prepared replay"));
    }
    shape
        .iter()
        .try_fold(1usize, |size, &dim| size.checked_mul(dim))
        .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| invalid("shape byte size overflow"))
}
pub(super) fn infer(operation: &Operation, shapes: &[&[usize]]) -> MlResult<Vec<usize>> {
    if operation.input_count() != Some(shapes.len()) || shapes.is_empty() {
        return Err(invalid("unsupported operation or input arity"));
    }
    let x = shapes[0];
    let output = match operation {
        Operation::Add | Operation::Sub | Operation::Mul | Operation::Div => {
            let y = shapes[1];
            let rank = x.len().max(y.len());
            let mut result = vec![1; rank];
            for offset in 0..rank {
                let a = x.len().checked_sub(offset + 1).map(|i| x[i]).unwrap_or(1);
                let b = y.len().checked_sub(offset + 1).map(|i| y[i]).unwrap_or(1);
                if a != b && a != 1 && b != 1 {
                    return Err(invalid("incompatible broadcast shapes"));
                }
                result[rank - 1 - offset] = a.max(b);
            }
            result
        }
        Operation::Matmul => {
            let y = shapes[1];
            if x.len() != 2 || y.len() != 2 || x[1] != y[0] {
                return Err(invalid(
                    "initial prepared Matmul requires compatible rank-2 matrices",
                ));
            }
            vec![x[0], y[1]]
        }
        Operation::Neg
        | Operation::Square
        | Operation::Exp
        | Operation::Log
        | Operation::Sqrt
        | Operation::Relu
        | Operation::Tanh
        | Operation::Sigmoid
        | Operation::Silu
        | Operation::Sin
        | Operation::Cos
        | Operation::Abs => x.to_vec(),
        Operation::Reshape(shape) => {
            if numel(shape)? != numel(x)? {
                return Err(invalid("reshape element count mismatch"));
            }
            shape.clone()
        }
        Operation::Transpose(axes) => {
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            if sorted != (0..x.len()).collect::<Vec<_>>() {
                return Err(invalid("invalid transpose permutation"));
            }
            axes.iter().map(|&axis| x[axis]).collect()
        }
        Operation::Sum => vec![],
        Operation::Loss {
            kind: LossKind::Mse | LossKind::Mae | LossKind::BinaryCrossEntropy,
            reduction: Reduction::Mean,
        } => {
            if x != shapes[1] {
                return Err(invalid("loss shape mismatch"));
            }
            vec![]
        }
        _ => {
            return Err(crate::MlError::UnsupportedCapability {
                module: "prepared replay",
                capability: "shape preparation for this operation",
                operation: operation.name(),
            });
        }
    };
    numel(&output)?;
    Ok(output)
}
impl ExecutionContext {
    /// Prepare without running kernels, reading tensor data, creating graph nodes,
    /// or consuming RNG. Parameter IDs/shapes fix the binding and alias signature.
    pub fn prepare(
        &self,
        program: &PreparedProgram,
        parameters: &[&Parameter],
        outputs: &[TensorSlotId],
        mode: PreparedMode,
    ) -> MlResult<PreparedPlan> {
        if self.route() != ExecutionRoute::P1 {
            return Err(invalid("preparation requires the P1 route"));
        }
        let provider = self
            .inner
            .operations
            .as_ref()
            .ok_or_else(|| invalid("operation provider unavailable"))?;
        if !provider.supports_prepared_replay() {
            return Err(crate::MlError::UnsupportedCapability {
                module: "prepared replay",
                capability: "provider opt-in",
                operation: "prepare",
            });
        }
        {
            let state = self
                .inner
                .state
                .try_borrow()
                .map_err(|_| crate::ContextError::BorrowConflict)?;
            state.store()?;
            if mode == PreparedMode::Training {
                state.engine()?;
            }
        }
        if outputs.is_empty()
            || outputs.iter().any(|id| {
                id.1 != program.id
                    || !program
                        .slots
                        .get(id.0)
                        .is_some_and(|slot| matches!(slot.source, Source::Node))
            })
        {
            return Err(invalid("invalid output slots"));
        }
        if parameters.len() != program.parameters.len() {
            return Err(invalid("parameter count mismatch"));
        }
        for (parameter, slot) in parameters.iter().zip(&program.parameters) {
            self.validate(parameter.tensor())?;
            if parameter.tensor().shape()? != program.slots[slot.0].shape {
                return Err(invalid("parameter shape mismatch"));
            }
        }
        Ok(PreparedPlan {
            context: self.id(),
            mode,
            program: program.clone(),
            outputs: outputs.to_vec(),
            parameter_ids: parameters.iter().map(|p| p.id()).collect(),
        })
    }
}
