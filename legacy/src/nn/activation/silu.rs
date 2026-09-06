use super::*;

impl Function for SiLUOp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(SiLUOp)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        // Numerically stable sigmoid:
        //   x >= 0 → σ = 1 / (1 + exp(-x))
        //   x <  0 → σ = exp(x) / (1 + exp(x))
        let sigmoid: Vec<f32> = x.data().iter().map(|&v| {
            if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let e = v.exp();
                e / (1.0 + e)
            }
        }).collect();
        // SiLU(x) = x * σ(x)
        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(x.data(), &sigmoid),
                x.shape()
            )?
        ])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        // Numerically stable sigmoid (same as forward)
        let sigmoid: Vec<f32> = x.data().iter().map(|&v| {
            if v >= 0.0 {
                1.0 / (1.0 + (-v).exp())
            } else {
                let e = v.exp();
                e / (1.0 + e)
            }
        }).collect();
        // σ'(x) = σ(x) * (1 - σ(x))
        let sigmoid_deriv: Vec<f32> = sigmoid.iter().map(|&s| s * (1.0 - s)).collect();
        // ∂SiLU/∂x = σ(x) + x * σ'(x)
        let derivative = self.backend.add(
            &sigmoid,
            &self.backend.multiply(x.data(), &sigmoid_deriv)
        );
        // ∂L/∂x = grad * ∂SiLU/∂x
        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(grad.data(), &derivative),
                grad.shape()
            )?
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}
