use super::*;

impl Function<f32> for Sigmoid {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sigmoid)
    }
    
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let x = Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| -x).collect(), targets[0].shape())?;
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
        let input_x = targets[0]; // 입력값 x

        //시그모이드 출력값 σ(x)
        let neg_x_data = input_x.data().iter().map(|&v| -v).collect::<Vec<f32>>();
        let ones = vec![1.0f32; input_x.data().len()];
        let exp_neg_x = self.backend.exp(&neg_x_data);
        let one_plus_exp = self.backend.add(&ones, &exp_neg_x);
        let sigmoid_output_data = self.backend.div(&ones, &one_plus_exp); // σ(x)
        
        // derivative = σ(x) * (1 - σ(x))
        let derivative = self.backend.multiply(
            &sigmoid_output_data,
            &self.backend.sub(
                &ones,
                &sigmoid_output_data
            )
        );

        // 최종 그래디언트: ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * derivative
        Ok(vec![
            Tensor::from_vec(
                self.backend.multiply(&grad.data(), &derivative),
                grad.shape()
            )?
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn node_id(&self) -> &NodeId { &self.node_id }
}