use super::*;

impl Function for SiLUOp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(SiLUOp)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        let ones = vec![1.0f32; x.data().len()];
        // σ(x) = 1 / (1 + exp(-x))
        let neg_x: Vec<f32> = x.data().iter().map(|&v| -v).collect();
        let sigmoid = self.backend.div(&ones, &self.backend.add(&ones, &self.backend.exp(&neg_x)));
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
        let ones = vec![1.0f32; x.data().len()];
        // σ(x) = 1 / (1 + exp(-x))
        let neg_x: Vec<f32> = x.data().iter().map(|&v| -v).collect();
        let sigmoid = self.backend.div(&ones, &self.backend.add(&ones, &self.backend.exp(&neg_x)));
        // σ'(x) = σ(x) * (1 - σ(x))
        let sigmoid_deriv = self.backend.multiply(&sigmoid, &self.backend.sub(&ones, &sigmoid));
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
