use super::*;

impl Activation<f32> for Sigmoid {
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!()
    }
    fn apply(&self, input: &Arc<Variable<f32>>) -> Arc<Variable<f32>> {
        unimplemented!()
    }
}

impl Function<f32> for Sigmoid {
    fn new() -> MlResult<Self> {
        Ok(
            Sigmoid {
                exp: Arc::new(Exp::new()?),

                #[cfg(all(feature = "enableBackpropagation"))]
                mul: Arc::new(Mul::new()?),
                #[cfg(all(feature = "enableBackpropagation"))]
                sub: Arc::new(Sub::new()?)
            }
        )
    }

    fn forward(&self, x: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        // σ(x) = 1/(1+e^(-x))
        Ok(vec![
            Tensor::from_vec(
                self.exp.forward(&[&-x[0]])?.remove(0).data().iter().map(|&e_neg_x| 1.0 / (1.0 + e_neg_x)).collect::<Vec<f32>>(),
                x[0].shape()
            )?]
        )
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let sigmoid_output = targets[0];
        // σ'(x) = σ(x) * (1 - σ(x))
        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * σ'(x)
        Ok(vec![
            self.mul.forward(&[
                grad, &self.mul.forward(&[
                    sigmoid_output,
                    &self.sub.forward(&[
                        &Tensor::from_vec(
                            vec![1.0; sigmoid_output.shape().iter().product()],
                            sigmoid_output.shape())?,
                        sigmoid_output
                    ])?.remove(0)
                ])?.remove(0)
            ])?.remove(0)
        ])
    }
}