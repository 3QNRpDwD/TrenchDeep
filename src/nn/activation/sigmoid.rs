use super::*;

impl Function for Sigmoid {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sigmoid)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        let ones = vec![1.0f32; x.data().len()];
        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.div(&ones, &self.backend.add(&ones, &self.backend.exp(x.data()))),
                x.shape()
            )?]
        )
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input_x = targets[0];
        let neg_x_data = input_x.data().iter().map(|&v| -v).collect::<Vec<f32>>();
        let ones = vec![1.0f32; input_x.data().len()];
        let exp_neg_x = self.backend.exp(&neg_x_data);
        let one_plus_exp = self.backend.add(&ones, &exp_neg_x);
        let sigmoid_output_data = self.backend.div(&ones, &one_plus_exp);

        let derivative = self.backend.multiply(
            &sigmoid_output_data,
            &self.backend.sub(&ones, &sigmoid_output_data)
        );

        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(&grad.data(), &derivative),
                grad.shape()
            )?
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    fn node_id(&self) -> &NodeId { &self.node_id }
}
