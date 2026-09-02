use super::*;

impl SoftmaxRegression {
    pub fn build_model(n_input: usize, n_output: usize) -> MlResult<SoftmaxRegression> {
        let activation = SoftmaxOp::new()?;
        let loss_function = SoftmaxCrossEntropyLoss::new()?;

        info!("Network Structure: {}(Input) -> {}(Output)", n_input, n_output);
        info!("Activation Functions: {} (Output)", activation.name());

        let sr = SoftmaxRegression::new(&[n_input, n_output], activation, loss_function);
        info!("softmax regression model created successfully.");
        Ok(sr)
    }

    pub fn new(
        layer_parms: &[usize],
        activation: GlobalFunction,
        loss_function: GlobalFunction,
    ) -> Self {
        let n_input = layer_parms[0];
        let n_output = layer_parms[1];
        let w1_data: Vec<f32> = (0..n_input * n_output)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_input, n_output]).unwrap(),
            "weight_1"
        );

        let b1_data: Vec<f32> = vec![0.0; n_output];
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_output]).unwrap(),
            "bias_1"
        );

        Self { w1, b1, activation, loss_function }
    }
}


impl Model for SoftmaxRegression {
    #[cfg(feature = "enableBackward")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;
        let uh1_pre = matmul.apply(&[x, &self.w1])?;
        Ok(&uh1_pre + &self.b1)
    }

    fn predict(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        let uh1_pre = matmul.forward(&[x, self.w1.tensor()])?.remove(0);
        let uh1 = add.forward(&[&uh1_pre, self.b1.tensor()])?.remove(0);
        let ah1 = self.activation.forward(&[&uh1])?.remove(0);

        Ok(ah1)
    }

}

#[cfg(feature = "enableBackward")]
impl crate::trainer::SupervisedModel for SoftmaxRegression {
    fn forward_loss(
        &mut self,
        x: &Variable,
        t: &Variable,
    ) -> MlResult<(Variable, Variable)> {
        let y = self.apply(x)?;
        let loss = self.loss_function.apply_with_label(&[&y, t], "loss")?;
        Ok((y, loss))
    }

    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<GlobalTensor<f32>> {
        self.predict(x)
    }
}

impl crate::trainer::TrainableModel for SoftmaxRegression {
    fn params(&self) -> Vec<&dyn crate::nn::Parameter> { vec![&self.w1, &self.b1] }
}
impl crate::trainer::CheckpointableModel for SoftmaxRegression {}
