use crate::contracts::*;
use crate::{AutogradError, MlResult};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default)]
pub struct ReverseMode {
    nodes: HashMap<TensorId, GradientRecord>,
}
impl AutogradEngine for ReverseMode {
    fn record(&mut self, record: GradientRecord) -> MlResult<()> {
        if self.nodes.contains_key(&record.output) {
            return Err(crate::TensorError::InvalidOperation {
                op: "record",
                reason: "duplicate graph output".into(),
            }
            .into());
        }
        self.nodes.insert(record.output, record);
        Ok(())
    }
    fn get(&self, output: TensorId) -> Option<GradientRecord> {
        self.nodes.get(&output).cloned()
    }
    fn remove(&mut self, output: TensorId) -> MlResult<Option<GradientRecord>> {
        Ok(self.nodes.remove(&output))
    }
    fn nodes(&self) -> Vec<TensorId> {
        self.nodes.keys().copied().collect()
    }
    fn order(&self, output: TensorId) -> MlResult<Vec<TensorId>> {
        // Iterative DFS also supports deep user graphs without overflowing the stack.
        let mut stack = vec![(output, false)];
        let mut visiting = HashSet::new();
        let mut finished = HashSet::new();
        let mut order = Vec::new();
        while let Some((id, exiting)) = stack.pop() {
            if finished.contains(&id) {
                continue;
            }
            let Some(node) = self.nodes.get(&id) else {
                finished.insert(id);
                continue;
            };
            if exiting {
                visiting.remove(&id);
                finished.insert(id);
                order.push(id);
                continue;
            }
            if !visiting.insert(id) {
                return Err(AutogradError::CycleDetected.into());
            }
            stack.push((id, true));
            for &input in node.inputs.iter().rev() {
                stack.push((input, false));
            }
        }
        Ok(order)
    }
}
