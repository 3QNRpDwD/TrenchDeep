use super::*;

impl Function for Sum {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sum)
    }
    
    fn forward(&self, input: &[&Tensor]) -> MlResult<Vec<Tensor>> {
        if input.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let result = input[0];
        for tensor in &input[1..] {
            if tensor.shape() != result.shape() {
                return Err(MlError::TensorError(TensorError::InvalidShape {
                    expected: result.shape().to_vec(),
                    got: tensor.shape().to_vec(),
                }));
            }
            result.data().iter_mut().zip(tensor.data().iter()).for_each(|(a, b)| *a += *b);        
        }

        Ok(vec![result.clone()])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, targets: &[&Tensor], grad: &Tensor) -> MlResult<Vec<Tensor>> {
        if targets.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }
        
        Ok(vec![grad.clone(); targets.len()])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}