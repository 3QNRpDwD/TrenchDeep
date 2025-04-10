use super::*;

impl Activation<f32> for Tanh {}
impl Function<f32> for Tanh {
    fn new() -> MlResult<Self> {
        Ok(
            Tanh {
                exp: Arc::new(Exp::new()?),
                sub: Arc::new(Sub::new()?),
                div: Arc::new(Div::new()?),
                add: Arc::new(Add::new()?),

                #[cfg(all(feature = "enableBackpropagation"))]
                mul: Arc::new(Mul::new()?),
            }
        )
    }

    fn forward(&self, x: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        // e^x
        let pos_exp = self.exp.forward(&[x[0]])?.remove(0);
        // e^(-x)
        let neg_exp = self.exp.forward(&[&-x[0]])?.remove(0);

        // (e^x - e^(-x)) / (e^x + e^(-x))
        Ok(vec![
            self.div.forward(&[
                &self.sub.forward(&[
                    &pos_exp,
                    &neg_exp
                ])?.remove(0),
                &self.add.forward(&[
                    &pos_exp,
                    &neg_exp
                ])?.remove(0)
            ])?.remove(0)
        ])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, target: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let tanh_output = target[0];
        // tanh^2(x)
        let tanh_squared = self.mul.forward(&[tanh_output, tanh_output])?.remove(0);

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * (1 - tanh^2(x))
        Ok(vec![
            self.mul.forward(&[
                grad,
                &self.sub.forward(&[
                    &Tensor::from_vec(
                        vec![1.0; tanh_squared.shape().iter().product()],
                        tanh_squared.shape()
                    )?,
                    &tanh_squared
                ])?.remove(0)
            ])?.remove(0)
        ])
    }
}