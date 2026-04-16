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
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, optimizer: &mut dyn crate::optimizer::Optimizer, tolerance: f32) -> MlResult<()> {
        crate::trainer::Trainer::default().fit(self, optimizer, x_set, t_set, epochs, tolerance)?;
        Ok(())
    }

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

    fn save(&self, path: &str) -> MlResult<()> {
        todo!()
    }

    fn load(&mut self, path: &str) -> MlResult<()> {
        todo!()
    }

    fn get_loss(&self) -> f32 {
        todo!()
    }

    fn compute_total_error(&mut self, x_set: &[&Variable], t_set: &[&Variable]) -> MlResult<f32> {
        let mut total_loss = 0.0;
        for m in 0..x_set.len() {
            let logit_tensor = {
                let matmul = Matmul::new()?;
                let add = Add::new()?;
                let uh1_pre = matmul.forward(&[x_set[m].tensor(), self.w1.tensor()])?.remove(0);
                add.forward(&[&uh1_pre, self.b1.tensor()])?.remove(0)
            };
            let loss = self.loss_function.forward(&[&logit_tensor, t_set[m].tensor()])?.remove(0);
            total_loss += loss.data()[0];
        }
        Ok(total_loss / x_set.len() as f32)
    }
}

#[cfg(feature = "enableBackward")]
impl crate::trainer::TrainableModel for SoftmaxRegression {
    fn forward_loss(
        &mut self,
        x: &Variable,
        t: &Variable,
    ) -> MlResult<(Variable, Variable)> {
        let y = self.apply(x)?;
        let loss = self.loss_function.apply_with_label(&[&y, t], "loss")?;
        Ok((y, loss))
    }

    fn params(&self) -> Vec<&dyn crate::nn::Parameter> {
        vec![&self.w1, &self.b1]
    }

    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<GlobalTensor<f32>> {
        self.predict(x)
    }
}
