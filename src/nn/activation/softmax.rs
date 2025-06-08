use super::*;

impl Function<f32> for Softmax {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Softmax)
    }
    
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let softmax_output = targets[0];
        let ones = vec![1.0f32; softmax_output.data().len()];
        
        // ∂L/∂x = ∂L/∂y * ∂y/∂x
        // The Jacobian of the softmax function is complex, but for simplicity, we can use:
        // ∂L/∂x = grad * (softmax_output * (1 - softmax_output))
        
        let softmax_grad = self.backend.multiply(
            &grad.data(),
            &self.backend.multiply(
                &softmax_output.data(),
                &self.backend.sub(&ones, &softmax_output.data())
            )
        );
        
        Ok(vec![Tensor::from_vec(softmax_grad, grad.shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn node_id(&self) -> &NodeId { &self.node_id }
}