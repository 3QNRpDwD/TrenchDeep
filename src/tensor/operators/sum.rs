use super::*;

impl Function<f32> for Sum {
    fn new() -> MlResult<Self> {
        Ok(Self { backend: Arc::new(CpuBackend::new()?) })
    }
    
    fn forward(&self, input: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        if input.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let mut result = input[0].clone();
        for tensor in &input[1..] {
            if tensor.shape != result.shape {
                return Err(MlError::TensorError(TensorError::InvalidShape {
                    expected: result.shape.clone(),
                    got: tensor.shape.clone(),
                }));
            }
            result.data.iter_mut().zip(tensor.data.iter()).for_each(|(a, b)| *a += b);
        }

        Ok(vec![result])
    }
}