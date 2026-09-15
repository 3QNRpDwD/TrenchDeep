use super::{invalid, plan::*};
use crate::{
    contracts::{Operation, OperationProvider, PreparedBackward},
    *,
};
use std::{collections::HashMap, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackwardPlanStats {
    pub roots: usize,
    pub nodes: usize,
    pub maximum_fan_in: usize,
    pub saved_buffers: usize,
}
#[derive(Debug)]
pub(super) struct Node {
    pub output: usize,
    pub inputs: Vec<usize>,
    pub vjp: Option<PreparedBackward>,
}
#[derive(Debug)]
pub(super) struct Root {
    pub slot: usize,
    pub order: Vec<usize>,
    pub fan_in: Vec<usize>,
}
#[derive(Debug)]
pub(crate) struct BackwardPlan {
    pub(super) nodes: Vec<Node>,
    pub(super) roots: Vec<Root>,
    pub(super) canonical: Vec<usize>,
    pub(super) tracked: Vec<bool>,
    pub(super) shapes: Vec<Vec<usize>>,
}
impl BackwardPlan {
    pub(super) fn stats(&self) -> BackwardPlanStats {
        BackwardPlanStats {
            roots: self.roots.len(),
            nodes: self.nodes.iter().filter(|n| n.vjp.is_some()).count(),
            maximum_fan_in: self
                .roots
                .iter()
                .flat_map(|r| &r.fan_in)
                .copied()
                .max()
                .unwrap_or(0),
            saved_buffers: self
                .nodes
                .iter()
                .filter_map(|n| n.vjp.as_ref())
                .map(|v| v.saved_shapes.len())
                .sum(),
        }
    }
}
pub(super) fn compile(
    program: &PreparedProgram,
    parameters: &[&Parameter],
    outputs: &[TensorSlotId],
    mode: PreparedMode,
    provider: &dyn OperationProvider,
) -> MlResult<BackwardPlan> {
    let count = program.slots.len();
    let mut plan = BackwardPlan {
        nodes: Vec::new(),
        roots: Vec::new(),
        canonical: (0..count).collect(),
        tracked: vec![false; count],
        shapes: program.slots.iter().map(|s| s.shape.clone()).collect(),
    };
    if mode == PreparedMode::Inference {
        return Ok(plan);
    }
    let mut aliases = HashMap::new();
    for (slot, parameter) in program.parameters.iter().zip(parameters) {
        plan.canonical[slot.0] = *aliases.entry(parameter.id()).or_insert(slot.0);
    }
    for (i, slot) in program.slots.iter().enumerate() {
        plan.tracked[i] = matches!(
            slot.source,
            Source::Parameter
                | Source::Feed {
                    requires_grad: true
                }
        );
    }
    let mut producer = vec![None; count];
    for (i, instruction) in program.instructions.iter().enumerate() {
        let inputs = instruction.inputs.iter().map(|s| s.0).collect::<Vec<_>>();
        let differentiable_inputs = if matches!(instruction.operation, Operation::Loss { .. }) {
            &inputs[..1]
        } else {
            &inputs[..]
        };
        let tracked = differentiable_inputs.iter().any(|&s| plan.tracked[s]);
        plan.tracked[instruction.output.0] = tracked;
        let vjp = if tracked {
            let shapes = inputs
                .iter()
                .map(|&s| plan.shapes[s].as_slice())
                .collect::<Vec<_>>();
            let vjp = provider
                .prepare_backward(&instruction.operation, &shapes)?
                .ok_or_else(|| invalid("provider does not support prepared backward"))?;
            if vjp.inputs.len() != inputs.len()
                || vjp.differentiable.len() != inputs.len()
                || vjp.operation.input_count() != inputs.len()
            {
                return Err(invalid("invalid prepared backward arity"));
            }
            for shape in &vjp.saved_shapes {
                super::prepare::numel(shape)?;
            }
            producer[instruction.output.0] = Some(i);
            Some(vjp)
        } else {
            None
        };
        plan.nodes.push(Node {
            output: instruction.output.0,
            inputs,
            vjp,
        });
    }
    for output in outputs {
        if !plan.tracked[output.0] {
            continue;
        }
        // Match the eager DFS ordering once, including fan-out, without consulting
        // runtime TensorIds, GradientRecords, or the autograd engine at run time.
        let mut stack = vec![(output.0, false)];
        let mut seen = vec![false; count];
        let mut order = Vec::new();
        while let Some((slot, exit)) = stack.pop() {
            if seen[slot] {
                continue;
            }
            let Some(index) = producer[slot] else {
                seen[slot] = true;
                continue;
            };
            if exit {
                seen[slot] = true;
                order.push(index);
                continue;
            }
            stack.push((slot, true));
            for &input in plan.nodes[index].inputs.iter().rev() {
                stack.push((input, false));
            }
        }
        let mut fan_in = vec![0; count];
        fan_in[plan.canonical[output.0]] = 1;
        // Only derivative-reachable nodes contribute. Loss targets stop gradients.
        let mut reachable = vec![false; count];
        reachable[output.0] = true;

        for &index in order.iter().rev() {
            let node = &plan.nodes[index];
            if !reachable[node.output] {
                continue;
            }
            let vjp = node.vjp.as_ref().unwrap();
            for (&slot, &enabled) in node.inputs.iter().zip(&vjp.differentiable) {
                if enabled && plan.tracked[slot] {
                    reachable[slot] = true;
                    fan_in[plan.canonical[slot]] += 1;
                }
            }
        }
        order.retain(|&index| reachable[plan.nodes[index].output]);
        plan.roots.push(Root {
            slot: output.0,
            order,
            fan_in,
        });
    }
    Ok(plan)
}

#[derive(Debug)]
pub(crate) struct PreparedRun {
    pub(super) plan: Rc<BackwardPlan>,
    pub(super) into: Option<Rc<std::cell::RefCell<super::into_executor::IntoStorage>>>,
    pub(super) values: Vec<TensorBuffer>,
    pub(super) saved: Vec<Vec<TensorBuffer>>,
    pub(super) ids: Vec<Option<TensorId>>,
    pub(super) exports: Vec<(TensorId, usize)>,
    pub(super) consumed: bool,
    pub(super) completed: bool,
}
impl ExecutionContext {
    pub(crate) fn reject_prepared_extension(&self) -> MlResult<()> {
        if self.inner.prepared_run.borrow().is_some() {
            return Err(invalid(
                "declare extra operations before preparation; prepared callbacks cannot extend the graph",
            ));
        }
        Ok(())
    }
    pub(crate) fn guard_prepared_update(&self) -> MlResult<()> {
        if self
            .inner
            .prepared_run
            .borrow()
            .as_ref()
            .is_some_and(|run| !run.completed)
        {
            return Err(invalid(
                "parameter updates require completed prepared backward",
            ));
        }
        Ok(())
    }
    pub(crate) fn backward_prepared(
        &self,
        output: &Variable,
        options: BackwardOptions<'_>,
        observe: impl FnOnce(&super::super::State) -> MlResult<()>,
    ) -> MlResult<()> {
        self.validate(output.tensor())?;
        let mut active = self
            .inner
            .prepared_run
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let run = active.as_mut().expect("active prepared run");
        let root_id = output.tensor().id();
        let slot = run
            .exports
            .iter()
            .find(|(id, _)| *id == root_id)
            .map(|(_, s)| *s)
            .ok_or_else(|| invalid("backward root must be a prepared export"))?;
        let root = run
            .plan
            .roots
            .iter()
            .find(|r| r.slot == slot)
            .ok_or(AutogradError::NodeNotFound(root_id))?;
        if run.consumed {
            return Err(AutogradError::GraphAlreadyFreed(root_id).into());
        }
        if options.retain_graph {
            return Err(invalid(
                "prepared replay currently permits one backward per run",
            ));
        }
        let shape = &run.plan.shapes[slot];
        let seed = if let Some(seed) = options.gradient {
            self.validate(seed)?;
            let buffer = seed.snapshot()?;
            if buffer.shape() != shape {
                return Err(AutogradError::GradientShapeMismatch {
                    expected: shape.clone(),
                    got: buffer.shape().to_vec(),
                }
                .into());
            }
            buffer
        } else {
            if super::prepare::numel(shape)? != 1 {
                return Err(AutogradError::OutputNotScalar(shape.clone()).into());
            }
            TensorBuffer::from_vec(vec![1.0], shape)?
        };
        if run.into.is_some() {
            return super::into_backward::execute(self, run, slot, seed, observe);
        }
        run.consumed = true;
        let mut gradients: Vec<Option<TensorBuffer>> = vec![None; run.values.len()];
        let mut received = vec![0usize; gradients.len()];
        let seed_slot = run.plan.canonical[slot];
        gradients[seed_slot] = Some(seed);
        received[seed_slot] = 1;
        for &index in root.order.iter().rev() {
            let node = &run.plan.nodes[index];
            let vjp = node.vjp.as_ref().unwrap();
            let Some(gradient) = gradients[node.output].as_ref() else {
                return Err(invalid("missing prepared cotangent"));
            };
            if received[node.output] != root.fan_in[node.output] {
                return Err(invalid("incomplete prepared gradient fan-in"));
            }
            let inputs = node
                .inputs
                .iter()
                .map(|&s| run.values[s].view())
                .collect::<Vec<_>>();
            let saved = run.saved[index]
                .iter()
                .map(TensorBuffer::view)
                .collect::<Vec<_>>();
            let incoming = vjp.operation.backward(&inputs, &saved, gradient.view())?;
            if incoming.len() != node.inputs.len() {
                return Err(invalid("prepared backward arity mismatch"));
            }
            for ((&input, value), &enabled) in
                node.inputs.iter().zip(&incoming).zip(&vjp.differentiable)
            {
                if let Some(value) = value {
                    if value.shape() != run.plan.shapes[input] {
                        return Err(invalid("prepared gradient shape mismatch"));
                    }
                }
                if enabled && run.plan.tracked[input] && value.is_none() {
                    return Err(invalid("missing declared input gradient"));
                }
            }
            for ((&input, value), &enabled) in
                node.inputs.iter().zip(incoming).zip(&vjp.differentiable)
            {
                if !enabled || !run.plan.tracked[input] {
                    continue;
                }
                let destination = run.plan.canonical[input];
                let value = value.unwrap();
                received[destination] += 1;
                if let Some(existing) = &mut gradients[destination] {
                    for (a, b) in existing.data.iter_mut().zip(value.data) {
                        *a += b;
                    }
                } else {
                    gradients[destination] = Some(value);
                }
            }
        }
        if received != root.fan_in {
            return Err(invalid("incomplete prepared gradient fan-in"));
        }
        let mut state = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        for (slot, id) in run.ids.iter().enumerate() {
            let Some(id) = id else {
                continue;
            };
            if run.plan.canonical[slot] != slot {
                continue;
            }
            if !state.leaves.contains(id) && !state.retained.contains(id) {
                continue;
            }
            if let Some(value) = gradients[slot].take() {
                if let Some(existing) = state.gradients.get_mut(id) {
                    for (a, b) in existing.data.iter_mut().zip(value.data) {
                        *a += b;
                    }
                } else {
                    state.gradients.insert(*id, value);
                }
            }
        }
        if let Err(error) = observe(&state) {
            state.gradients.clear();
            return Err(error);
        }
        state.consumed.extend(run.exports.iter().map(|(id, _)| *id));
        run.completed = true;
        Ok(())
    }
}
