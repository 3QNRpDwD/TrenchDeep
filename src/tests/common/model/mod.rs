use super::*;
pub mod mlp;
pub mod regression;

pub trait Model {
    #[cfg(feature = "enableBackward")]
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()>;
    #[cfg(feature = "enableBackward")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable>;
    fn predict(&mut self, test_data: &dyn TensorBase) -> MlResult<GlobalTensor<f32>>;
    #[cfg(feature = "enableBackward")]
    fn update(&mut self, lr: &dyn TensorBase) -> MlResult<()>;
    #[cfg(feature = "enableBackward")]
    fn zero_grad(&mut self) -> MlResult<()>;
    fn save(&self, path: &str) -> MlResult<()>;
    fn load(&mut self, path: &str) -> MlResult<()>;
    fn get_loss(&self) -> f32;
    fn compute_total_error(&mut self, X: &[&Variable], T: &[&Variable]) -> MlResult<f32>;
    fn evaluate_model(&mut self, x_test: &[&Variable], t_test: &[&Variable]) -> MlResult<f32> {
        let n_val = x_test.len();
        // info!("Starting model evaluation on {} test samples...", n_val);

        let mut correct_predictions = 0;
        for i in 0..n_val {
            let test_input = &x_test[i];
            let true_label_tensor = &t_test[i];

            let y = self.predict(test_input.tensor())?;

            // 예측 결과에서 가장 확률이 높은 클래스의 인덱스를 찾습니다.
            let predicted_class = y.data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);

            // 실제 레이블(one-hot)에서 정답 클래스의 인덱스를 찾습니다.
            // let true_class = true_label_tensor.tensor().data()
            //     .iter()
            //     .position(|&r| r == 1.0)
            //     .unwrap_or(0);
            let true_class = true_label_tensor.tensor().data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);

            if predicted_class == true_class {
                correct_predictions += 1;
            }
        }

        let accuracy = correct_predictions as f32 / n_val as f32 * 100.0;
        // info!("✅ Evaluation complete: Accuracy = {:.2}%", accuracy);

        Ok(accuracy)
    }
}

pub struct MLP {
    pub w1: Variable, // shape = [hidden_node, input_node]
    pub w2: Variable, // shape = [output_node, hidden_node]
    pub b1: Variable, // shape = [hidden_node, 1]
    pub b2: Variable, // shape = [output_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    layer: Sequential,
    loss_function: GlobalFunction,
}

pub struct SoftmaxRegression {
    pub w1: Variable, // shape = [hidden_node, input_node]
    pub b1: Variable, // shape = [hidden_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    activation: GlobalFunction,
    loss_function: GlobalFunction,
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  w1.shape = {:?}, w2.shape = {:?}",
                 self.w1.tensor().shape(),

                 self.w2.tensor().shape())?;
        // 활성화 함수 정보 추가
        writeln!(f, "  layer = {:?}", self.layer)?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}