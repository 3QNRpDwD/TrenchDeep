use super::*;

impl MLP {
    pub fn build_model(n_input: usize, n_hidden: usize, n_output: usize) -> MlResult<MLP> {
        let loss_function = CrossEntropyLoss::new()?;
        info!("Network Structure: {}(Input) -> {}(Hidden) -> {}(Output)", n_input, n_hidden, n_output);
        let mlp = MLP::new(
            &[n_input, n_hidden, n_output],
            Box::new(crate::nn::activation::Sigmoid::new("hidden_act")?),
            Box::new(crate::nn::activation::Softmax::new("output_act")?),
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
    fn apply(&mut self, x: &Variable) -> MlResult<Variable> {
        self.layer.apply(x)
    }

    fn predict(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        self.layer.predict(x)
    }

}

#[cfg(feature = "enableBackward")]
impl crate::trainer::SupervisedModel for MLP {
    fn forward_loss(
        &mut self,
        x: &Variable,
        t: &Variable,
    ) -> MlResult<(Variable, Variable)> {
        let y = self.layer.apply(x)?;
        let loss = self.loss_function.apply_with_label(&[&y, t], "loss")?;
        Ok((y, loss))
    }

    fn predict_raw(
        &mut self,
        x: &dyn TensorBase,
    ) -> MlResult<GlobalTensor<f32>> {
        self.layer.predict(x)
    }
}

impl crate::trainer::TrainableModel for MLP {
    fn params(&self) -> Vec<&dyn Parameter> { self.layer.params() }
}
impl crate::trainer::CheckpointableModel for MLP {
    fn save_checkpoint(&self, path: &std::path::Path) -> MlResult<()> {
        self.layer.save(path.to_string_lossy().as_ref())
    }
    fn load_checkpoint(&mut self, path: &std::path::Path) -> MlResult<()> {
        self.layer.load(path.to_string_lossy().as_ref())
    }
}
