use trench_deep::{AutogradError, ContextError, MlError, MlResult, TensorError, contracts::*};

/// Slot storage deliberately differs from the built-in hash map of Rc buffers.
#[derive(Debug, Default)]
pub struct SlotStore {
    names: Vec<(TensorId, usize)>,
    slots: Vec<Option<TensorBuffer>>,
}
impl SlotStore {
    fn slot(&self, id: TensorId) -> MlResult<usize> {
        self.names
            .iter()
            .find(|(key, _)| *key == id)
            .map(|(_, slot)| *slot)
            .ok_or_else(|| ContextError::UnknownTensor(id).into())
    }
}
impl TensorStore for SlotStore {
    fn insert(&mut self, id: TensorId, value: TensorBuffer) -> MlResult<()> {
        if self.names.iter().any(|(key, _)| *key == id) {
            return Err(MlError::StringError("duplicate ID".into()));
        }
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.slots.len());
        if slot == self.slots.len() {
            self.slots.push(Some(value));
        } else {
            self.slots[slot] = Some(value);
        }
        self.names.push((id, slot));
        Ok(())
    }
    fn alias(&mut self, id: TensorId, source: TensorId) -> MlResult<()> {
        let slot = self.slot(source)?;
        self.names.push((id, slot));
        Ok(())
    }
    fn with_view(
        &self,
        id: TensorId,
        f: &mut dyn FnMut(TensorView<'_>) -> MlResult<()>,
    ) -> MlResult<()> {
        let slot = self.slot(id)?;
        f(self.slots[slot]
            .as_ref()
            .ok_or(ContextError::UnknownTensor(id))?
            .view())
    }
    fn replace(&mut self, id: TensorId, value: TensorBuffer) -> MlResult<()> {
        let slot = self.slot(id)?;
        let old = self.slots[slot]
            .as_ref()
            .ok_or(ContextError::UnknownTensor(id))?;
        if old.shape() != value.shape() {
            return Err(TensorError::InvalidShape {
                expected: old.shape().to_vec(),
                got: value.shape().to_vec(),
            }
            .into());
        }
        self.slots[slot] = Some(value);
        Ok(())
    }
    fn remove(&mut self, id: TensorId) -> MlResult<()> {
        if let Some(index) = self.names.iter().position(|(key, _)| *key == id) {
            let (_, slot) = self.names.remove(index);
            if !self.names.iter().any(|(_, s)| *s == slot) {
                self.slots[slot] = None;
            }
        }
        Ok(())
    }
}

/// An insertion tape with Kahn traversal, independent of the default DFS graph.
#[derive(Debug, Default)]
pub struct Tape {
    records: Vec<GradientRecord>,
}
impl AutogradEngine for Tape {
    fn record(&mut self, node: GradientRecord) -> MlResult<()> {
        if self.get(node.output).is_some() {
            return Err(MlError::StringError("duplicate output".into()));
        }
        self.records.push(node);
        Ok(())
    }
    fn get(&self, id: TensorId) -> Option<GradientRecord> {
        self.records.iter().find(|n| n.output == id).cloned()
    }
    fn remove(&mut self, id: TensorId) -> MlResult<Option<GradientRecord>> {
        Ok(self
            .records
            .iter()
            .position(|n| n.output == id)
            .map(|i| self.records.remove(i)))
    }
    fn nodes(&self) -> Vec<TensorId> {
        self.records.iter().map(|n| n.output).collect()
    }
    fn order(&self, root: TensorId) -> MlResult<Vec<TensorId>> {
        let mut reachable = vec![root];
        let mut i = 0;
        while i < reachable.len() {
            if let Some(n) = self.get(reachable[i]) {
                for input in n.inputs {
                    if !reachable.contains(&input) {
                        reachable.push(input);
                    }
                }
            }
            i += 1;
        }
        let mut pending = self
            .records
            .iter()
            .filter(|n| reachable.contains(&n.output))
            .cloned()
            .collect::<Vec<_>>();
        let mut order = Vec::new();
        while !pending.is_empty() {
            let index = pending
                .iter()
                .position(|n| {
                    n.inputs
                        .iter()
                        .all(|input| !pending.iter().any(|p| p.output == *input))
                })
                .ok_or(AutogradError::CycleDetected)?;
            order.push(pending.remove(index).output);
        }
        Ok(order)
    }
}

#[derive(Debug, Default)]
pub struct ReferenceOps;
#[derive(Debug)]
struct Vjp {
    operation: Operation,
}
impl OperationProvider for ReferenceOps {
    fn execute(&self, op: &Operation, x: &[TensorView<'_>]) -> MlResult<OperationOutput> {
        let output = match op {
            Operation::Add | Operation::Mul => {
                let size = x[0].len().max(x[1].len());
                if x[0].len() != x[1].len() && x[1].len() != 1 && x[0].len() != 1 {
                    return Err(MlError::StringError("reference shape mismatch".into()));
                }
                let shape = if x[0].len() == size {
                    x[0].shape()
                } else {
                    x[1].shape()
                };
                TensorBuffer::from_vec(
                    (0..size)
                        .map(|i| {
                            let a = x[0].data()[i % x[0].len()];
                            let b = x[1].data()[i % x[1].len()];
                            if matches!(op, Operation::Add) {
                                a + b
                            } else {
                                a * b
                            }
                        })
                        .collect(),
                    shape,
                )?
            }
            Operation::Square => {
                TensorBuffer::from_vec(x[0].data().iter().map(|x| x * x).collect(), x[0].shape())?
            }
            Operation::Sum => TensorBuffer::from_vec(vec![x[0].data().iter().sum()], &[])?,
            Operation::Matmul => {
                let [m, k] = x[0].shape() else {
                    return Err(MlError::StringError("rank".into()));
                };
                let [kk, n] = x[1].shape() else {
                    return Err(MlError::StringError("rank".into()));
                };
                if k != kk {
                    return Err(MlError::StringError("inner dimension".into()));
                }
                let mut data = vec![0.0; m * n];
                for r in 0..*m {
                    for c in 0..*n {
                        data[r * n + c] = (0..*k)
                            .map(|p| x[0].data()[r * k + p] * x[1].data()[p * n + c])
                            .sum();
                    }
                }
                TensorBuffer::from_vec(data, &[*m, *n])?
            }
            Operation::Loss {
                kind: LossKind::Mse,
                reduction: Reduction::Mean,
            } => {
                let loss = x[0]
                    .data()
                    .iter()
                    .zip(x[1].data())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    / x[0].len() as f32;
                TensorBuffer::from_vec(vec![loss], &[])?
            }
            _ => {
                return Err(MlError::UnsupportedCapability {
                    module: "reference operations",
                    capability: "operation",
                    operation: op.name(),
                });
            }
        };
        Ok(OperationOutput {
            outputs: vec![output],
            saved: vec![],
            backward: Some(Box::new(Vjp {
                operation: op.clone(),
            })),
        })
    }
}
impl BackwardOp for Vjp {
    fn name(&self) -> &'static str {
        self.operation.name()
    }
    fn input_count(&self) -> usize {
        self.operation.input_count().unwrap_or(1)
    }
    fn backward(
        &self,
        x: &[TensorView<'_>],
        _: &[TensorView<'_>],
        g: TensorView<'_>,
    ) -> MlResult<Vec<Option<TensorBuffer>>> {
        let mut gradients = x.iter().map(|v| vec![0.0; v.len()]).collect::<Vec<_>>();
        match &self.operation {
            Operation::Add | Operation::Mul => {
                for (i, &g) in g.data().iter().enumerate() {
                    gradients[0][i % x[0].len()] += g * if matches!(self.operation, Operation::Mul)
                    {
                        x[1].data()[i % x[1].len()]
                    } else {
                        1.0
                    };
                    gradients[1][i % x[1].len()] += g * if matches!(self.operation, Operation::Mul)
                    {
                        x[0].data()[i % x[0].len()]
                    } else {
                        1.0
                    };
                }
            }
            Operation::Square => {
                for (i, &g) in g.data().iter().enumerate() {
                    gradients[0][i] = 2.0 * x[0].data()[i] * g;
                }
            }
            Operation::Sum => gradients[0].fill(g.data()[0]),
            Operation::Loss { .. } => {
                for i in 0..x[0].len() {
                    gradients[0][i] =
                        2.0 * (x[0].data()[i] - x[1].data()[i]) * g.data()[0] / x[0].len() as f32;
                }
                return Ok(vec![
                    Some(TensorBuffer::from_vec(gradients.remove(0), x[0].shape())?),
                    None,
                ]);
            }
            Operation::Matmul => {
                let m = x[0].shape()[0];
                let k = x[0].shape()[1];
                let n = x[1].shape()[1];
                for r in 0..m {
                    for p in 0..k {
                        for c in 0..n {
                            gradients[0][r * k + p] += g.data()[r * n + c] * x[1].data()[p * n + c];
                            gradients[1][p * n + c] += x[0].data()[r * k + p] * g.data()[r * n + c];
                        }
                    }
                }
            }
            _ => return Err(AutogradError::BackwardNotSupported(self.name().into()).into()),
        }
        gradients
            .into_iter()
            .zip(x)
            .map(|(g, x)| TensorBuffer::from_vec(g, x.shape()).map(Some))
            .collect()
    }
}
