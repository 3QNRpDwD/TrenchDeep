use super::*;

impl Activation<f32> for Sigmoid {
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!()
    }
    fn apply(&mut self, input: &Arc<Variable<f32>>) -> MlResult<Variable<f32>> {
        self.apply(input)
    }
}

impl Function<f32> for Sigmoid {
    fn new() -> MlResult<Self> { Ok(Sigmoid { backend: Arc::new(CpuBackend::new()?) }) }

    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let x = targets[0];
        let ones = vec![1.0f32; x.data().len()];
        Ok(vec![
            Tensor::from_vec(
                self.backend.div(&ones, &self.backend.add(&ones, &self.backend.exp(x.data()))),
                x.shape()
            )?]
        )
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let sigmoid_output = targets[0];
        // σ'(x) = σ(x) * (1 - σ(x))
        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * σ'(x)

        Ok(vec![
            Tensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &self.backend.multiply(
                        &sigmoid_output.data(),
                        &self.backend.sub(
                            &vec![1.0f32; sigmoid_output.data().len()],
                            &sigmoid_output.data()
                        )
                    )
                ),
                grad.shape()
            )?
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}