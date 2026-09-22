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
    if operation
        .input_count()
        .is_some_and(|count| count != shapes.len())
        || shapes.is_empty()
    {
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
            if x.len() < 2 || y.len() < 2 || x[x.len() - 1] != y[y.len() - 2] {
                return Err(invalid(
                    "prepared Matmul requires compatible matrices of rank >= 2",
                ));
            }
            let mut shape = infer(&Operation::Add, &[&x[..x.len() - 2], &y[..y.len() - 2]])?;
            shape.extend_from_slice(&[x[x.len() - 2], y[y.len() - 1]]);
            shape
        }
        Operation::Concat { axis } => {
            if *axis >= x.len()
                || shapes.iter().any(|s| {
                    s.len() != x.len()
                        || s.iter()
                            .zip(x)
                            .enumerate()
                            .any(|(i, (a, b))| i != *axis && a != b)
                })
            {
                return Err(invalid("invalid concat shapes or axis"));
            }
            let mut result = x.to_vec();
            result[*axis] = shapes
                .iter()
                .try_fold(0usize, |sum, s| sum.checked_add(s[*axis]))
                .ok_or_else(|| invalid("concat overflow"))?;
            result
        }
        Operation::Softmax { axis } => {
            if *axis >= x.len() {
                return Err(invalid("invalid softmax axis"));
            }
            x.to_vec()
        }
        Operation::Conv2d { stride, padding } => {
            let k = shapes[1];
            if x.len() != 4
                || k.len() != 4
                || x[1] != k[1]
                || shapes[2] != [k[0]]
                || stride.0 == 0
                || stride.1 == 0
            {
                return Err(invalid("invalid Conv2d shape/stride"));
            }
            let dimension = |input: usize, kernel: usize, stride: usize, pad: usize| {
                pad.checked_mul(2)
                    .and_then(|p| input.checked_add(p))
                    .and_then(|n| n.checked_sub(kernel))
                    .and_then(|n| (n / stride).checked_add(1))
                    .ok_or_else(|| invalid("Conv2d spatial overflow or kernel too large"))
            };
            vec![
                x[0],
                k[0],
                dimension(x[2], k[2], stride.0, padding.0)?,
                dimension(x[3], k[3], stride.1, padding.1)?,
            ]
        }
        Operation::GroupNorm { groups, epsilon } => {
            if x.len() != 4
                || *groups == 0
                || x[1] % groups != 0
                || shapes[1] != [x[1]]
                || shapes[2] != [x[1]]
                || !epsilon.is_finite()
                || *epsilon <= 0.0
            {
                return Err(invalid("invalid GroupNorm shape/attributes"));
            }
            x.to_vec()
        }
        Operation::NearestUpsample2d { scale } => {
            if x.len() != 4 || scale.0 == 0 || scale.1 == 0 {
                return Err(invalid("invalid upsample shape/scale"));
            }
            vec![
                x[0],
                x[1],
                x[2].checked_mul(scale.0)
                    .ok_or_else(|| invalid("upsample overflow"))?,
                x[3].checked_mul(scale.1)
                    .ok_or_else(|| invalid("upsample overflow"))?,
            ]
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
        Operation::Loss { kind, reduction } => {
            if x != shapes[1] {
                return Err(invalid("loss shape mismatch"));
            }
            if matches!(kind, LossKind::Huber { delta } if !delta.is_finite() || *delta <= 0.0) {
                return Err(crate::LossError::InvalidOperation {
                    op: "huber_loss",
                    reason: "delta must be finite and positive".into(),
                }
                .into());
            }
            let categorical =
                matches!(kind, LossKind::CrossEntropy | LossKind::SoftmaxCrossEntropy);
            if categorical && x.is_empty() {
                return Err(invalid("categorical loss requires a class axis"));
            }
            match reduction {
                Reduction::Mean | Reduction::Sum => vec![],
                Reduction::None if categorical => x[..x.len() - 1].to_vec(),
                Reduction::None => x.to_vec(),
            }
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
        let mut roots = Vec::new();
        if mode == PreparedMode::Training {
            for &output in outputs {
                if !roots.contains(&output) {
                    roots.push(output);
                }
            }
        }
        self.prepare_with_roots(program, parameters, outputs, &roots, mode)
    }

    /// Export all `outputs`, but allow backward only from the selected output
    /// slots. Other exported intermediates can still retain gradients reached
    /// from a selected root. The original `prepare` selects all training outputs.
    pub fn prepare_with_roots(
        &self,
        program: &PreparedProgram,
        parameters: &[&Parameter],
        outputs: &[TensorSlotId],
        backward_roots: &[TensorSlotId],
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
        if (mode == PreparedMode::Inference && !backward_roots.is_empty())
            || backward_roots
                .iter()
                .enumerate()
                .any(|(i, root)| !outputs.contains(root) || backward_roots[..i].contains(root))
        {
            return Err(invalid(
                "backward roots must be distinct exported training outputs",
            ));
        }
        for (parameter, slot) in parameters.iter().zip(&program.parameters) {
            self.validate(parameter.tensor())?;
            if parameter.tensor().shape()? != program.slots[slot.0].shape {
                return Err(invalid("parameter shape mismatch"));
            }
        }
        let backward = std::rc::Rc::new(super::backward::compile(
            program,
            parameters,
            backward_roots,
            mode,
            provider.as_ref(),
        )?);
        let buffers = super::buffers::compile(program, parameters, outputs, &backward)?;
        Ok(PreparedPlan {
            input_signature: None,
            backward,
            buffers,
            context: self.id(),
            mode,
            program: program.clone(),
            outputs: outputs.to_vec(),
            parameter_ids: parameters.iter().map(|p| p.id()).collect(),
        })
    }
}
