use super::*;

impl MLP {
    pub fn build_model(n_input: usize, n_hidden: usize, n_output: usize) -> MlResult<MLP> {
        let loss_function = CrossEntropyLoss::new()?;
        info!("Network Structure: {}(Input) -> {}(Hidden) -> {}(Output)", n_input, n_hidden, n_output);
        let mlp = MLP::new(
            &[n_input, n_hidden, n_output],
            Box::new(crate::nn::activation::SigmoidLayer::new("hidden_act")),
            Box::new(crate::nn::activation::SoftmaxLayer::new("output_act")),
            loss_function,
        )?;
        info!("MLP model created successfully.");
        Ok(mlp)
    }

    pub fn new(
        layer_params: &[usize], // [n_input, n_hidden, n_output]
        hidden_act: Box<dyn Layer>,
        output_act: Box<dyn Layer>,
        loss_function: GlobalFunction,
    ) -> MlResult<Self> {
        let layer = Sequential::from(vec![
            Box::new(crate::nn::Linear::new(layer_params[0], layer_params[1], "linear1")?),
            hidden_act,
            Box::new(crate::nn::Linear::new(layer_params[1], layer_params[2], "linear2")?),
            output_act,
        ], "MLP");
        Ok(Self { layer, loss_function })
    }
}

impl Model for MLP {
    #[cfg(feature = "enableBackward")]
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, optimizer: &mut dyn crate::optimizer::Optimizer, tolerance: f32) -> MlResult<()> {
        for param in self.layer.params() {
            optimizer.register(param);
        }
        crate::trainer::Trainer::default().fit(self, optimizer, x_set, t_set, epochs, tolerance)?;
        Ok(())
    }

    #[cfg(feature = "enableBackward")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable> {
        self.layer.apply(x)
    }

    fn predict(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        self.layer.predict(x)
    }

    fn save(&self, path: &str) -> MlResult<()> {
        self.layer.save(path)
    }

    fn load(&mut self, path: &str) -> MlResult<()> {
        self.layer.load(path)
    }

    fn get_loss(&self) -> f32 {
        todo!()
    }

    fn compute_total_error(&mut self, x_set: &[&Variable], t_set: &[&Variable]) -> MlResult<f32> {
        let mut total_loss = 0.0;
        for m in 0..x_set.len() {
            let y = self.predict(x_set[m].tensor())?;
            let loss = self.loss_function.forward(&[&y, t_set[m].tensor()])?.remove(0);
            total_loss += loss.data()[0];
        }
        Ok(total_loss / x_set.len() as f32)
    }
}

#[cfg(feature = "enableBackward")]
impl crate::trainer::TrainableModel for MLP {
    fn forward_loss(
        &mut self,
        x: &Variable,
        t: &Variable,
    ) -> MlResult<(Variable, Variable)> {
        let y = self.layer.apply(x)?;
        let loss = self.loss_function.apply_with_label(&[&y, t], "loss")?;
        Ok((y, loss))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        self.layer.params()
    }

    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<GlobalTensor<f32>> {
        self.layer.predict(x)
    }
}
