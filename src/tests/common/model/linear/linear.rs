use super::*;

impl LinearRegression {
    pub fn build_model(n_input: usize, n_output: usize) -> MlResult<LinearRegression> {
        let activation = IdentityOp::new()?;
        let loss_function = crate::loss::MeanSquaredError::new()?;

        info!("Network Structure: {}(Input) -> {}(Output)", n_input, n_output);
        info!("Activation Functions: {} (Output)", activation.name());

        let sr = LinearRegression::new(&[n_input, n_output], activation, loss_function);
        info!("linear regression model created successfully.");
        Ok(sr)
    }

    pub fn new(
        layer_parms: &[usize],
        activation: GlobalFunction,
        loss_function: GlobalFunction,
    ) -> Self {
        let n_input = layer_parms[0];
        let n_output = layer_parms[1];
        // He 초기화 또는 Xavier 초기화와 같은 더 나은 가중치 초기화 방법을 고려할 수 있음
        let w1_data: Vec<f32> = (0..n_input * n_output)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_input, n_output]).unwrap(),
            "weight_1"
        );

        // bias: [n_output] — Add의 broadcasting으로 [1, n_output] + [n_output] 지원
        let b1_data: Vec<f32> = vec![0.0; n_output];
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_output]).unwrap(),
            "bias_1"
        );

        Self { w1, b1, activation, loss_function }
    }
}


impl Model for LinearRegression {
    #[cfg(feature = "enableBackward")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;
        // x [1, n_input] × w1 [n_input, n_output] = [1, n_output]
        let linear_out = &matmul.apply(&[x, &self.w1])? + &self.b1;
        self.activation.apply(&[&linear_out])
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
impl crate::trainer::SupervisedModel for LinearRegression {
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

impl crate::trainer::TrainableModel for LinearRegression {
    fn params(&self) -> Vec<&dyn Parameter> { vec![&self.w1, &self.b1] }
}
impl crate::trainer::CheckpointableModel for LinearRegression {}

#[cfg(all(test, feature = "enableBackward"))]
mod data_loader_tests {
    use super::*;
    use crate::{
        optimizer::{Optimizer, SGD},
        trainer::{
            DataLoader, DatasetBuilder, EpochSchedule, MemorySource, SupervisedSample,
            SupervisedStackCollator, Trainer,
        },
    };

    #[test]
    fn supervised_trainer_accepts_built_loader() -> MlResult<()> {
        let dataset = DatasetBuilder::from_source(MemorySource::new(vec![
            ([0.0_f32, 0.0], [0.0_f32]),
            ([1.0, 0.0], [1.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 1.0], [2.0]),
        ]))
        .map(|(input, target): ([f32; 2], [f32; 1])| {
            Ok(SupervisedSample::new(
                Tensor::from_vec(input.to_vec(), &[2])?,
                Tensor::from_vec(target.to_vec(), &[1])?,
            ))
        })
        .build()?;
        let mut loader = DataLoader::builder(dataset)
            .collator(SupervisedStackCollator::new())
            .batch_size(2)
            .shuffle(false)
            .build()?;

        let mut model = LinearRegression::build_model(2, 1)?;
        let mut optimizer = SGD::new(0.01);
        for parameter in crate::trainer::TrainableModel::params(&model) {
            optimizer.register(parameter);
        }

        let result = Trainer::silent().supervised().fit(
            &mut model,
            &mut optimizer,
            &mut loader,
            EpochSchedule::new(2)?,
        )?;
        assert_eq!(result.units_completed, 2);
        assert!(result.final_loss.is_finite());
        Ok(())
    }
}
