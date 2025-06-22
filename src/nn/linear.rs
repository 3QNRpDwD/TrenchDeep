use super::*;

impl Layer for Linear {
    fn forward(&self, input: Arc<Variable>) -> MlResult<Arc<Variable>> {
        todo!()
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        todo!()
    }

    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    fn label(&self) -> &str {
        &self.label
    }
}