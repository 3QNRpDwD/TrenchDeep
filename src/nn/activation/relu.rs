use super::*;

impl Activation<f32> for Relu {
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!()
    }
    fn apply(&self, input: &Arc<Variable<f32>>) -> Arc<Variable<f32>> {
        unimplemented!()
    }
}

impl Function<f32> for Relu {
    fn new() -> MlResult<Self> {
        Ok(
            Relu {
                #[cfg(all(feature = "enableBackpropagation"))]
                mul: Arc::new(Mul::new()?)
            }
        )
    }

    fn forward(&self, x: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        // ReLU(x) = max(0, x)
        let result = x[0].data().iter()
            .map(|&val| if val > 0.0 { val } else { 0.0 })
            .collect::<Vec<f32>>();

        Ok(vec![Tensor::from_vec(result, x[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, target: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let relu_output = target[0];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * mask
        Ok(vec![
            self.mul.forward(&[
                grad,
                &Tensor::from_vec(
                    relu_output.data().iter()
                        .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                        .collect::<Vec<f32>>(),
                    relu_output.shape()
                )?
            ])?.remove(0)])
    }
}