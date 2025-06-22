use super::*;

impl Layer for Linear {
    fn forward(&mut self, input: Arc<Variable>) -> MlResult<Arc<Variable>> {
        todo!()
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }

    fn inputs_cache(&self) -> &HashSet<NodeId> {
        todo!()
    }

    fn outputs_cache(&self) -> &HashMap<NodeId, NodeId> {
        todo!()
    }

    fn inputs_cache_mut(&mut self) -> &mut HashSet<NodeId> {
        todo!()
    }

    fn outputs_cache_mut(&mut self) -> &mut HashMap<NodeId, NodeId> {
        todo!()
    }

    fn label(&self) -> &str {
        &self.label
    }
}