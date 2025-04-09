use std::sync::Arc;
use crate::MlResult;
use crate::nn::activation::{Relu, Sigmoid, Tanh};
use crate::tensor::operators::{Add, Div, Exp, Function, Mul, Sub};
use crate::tensor::{Tensor, TensorBase};

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