use super::*;

impl Layer for Pooling {
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        todo!()
    }
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        todo!()
    }
    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }
    fn inputs_cache(&self) -> &HashMap<NodeId, NodeId> { todo!() }
    fn inputs_cache_mut(&mut self) -> &mut HashMap<NodeId, NodeId> { todo!() }
    fn label(&self) -> &str {
        &self.label
    }
}