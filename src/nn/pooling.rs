use super::*;

impl MaxPooling {
    pub fn new(label: &str, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> MlResult<Self> {
        Ok(Self {
            label: label.to_string(),
            max_pool: MaxPool::new(kernel_size, stride, padding),
        })
    }
}

impl Layer for MaxPooling {
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let output = self.max_pool.forward(&[input.tensor()])?.remove(0);
        let var_act = var_act!(output.to_id(true)?, self.label());
        let op: Arc<dyn Function + Send + Sync> = self.max_pool.clone();
        var_act.with_grad_fn(op, &[input]);
        Ok(var_act)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let pooled_tensor = self.max_pool.forward(&[input])?.remove(0);
        TENSOR_ALLOCATOR.with_borrow(|alloc| {
            Ok(alloc.get_tensor_ref(&pooled_tensor.id()).unwrap().clone())
        })
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl AvgPooling {
    pub fn new(label: &str, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> MlResult<Self> {
        Ok(Self {
            label: label.to_string(),
            avg_pool: AvgPool::new(kernel_size, stride, padding),
        })
    }
}

impl Layer for AvgPooling {
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let output = self.avg_pool.forward(&[input.tensor()])?.remove(0);
        let var_act = var_act!(output.to_id(true)?, self.label());
        let op: Arc<dyn Function + Send + Sync> = self.avg_pool.clone();
        var_act.with_grad_fn(op, &[input]);
        Ok(var_act)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let pooled_tensor = self.avg_pool.forward(&[input])?.remove(0);
        TENSOR_ALLOCATOR.with_borrow(|alloc| {
            Ok(alloc.get_tensor_ref(&pooled_tensor.id()).unwrap().clone())
        })
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }

    fn label(&self) -> &str {
        &self.label
    }
}
