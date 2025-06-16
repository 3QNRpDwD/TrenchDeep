use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use crate::loss::{HuberLoss, MeanSquaredError};
use crate::nn::activation::ReLu;
use super::*;

pub fn build_model(n_input: usize, n_hidden: usize, n_output: usize) -> MlResult<MLP> {
    let hidden_activation = Sigmoid::new()?;
    let output_activation = Softmax::new()?;
    let loss_function = HuberLoss::new()?;

    info!("Network Structure: {}(Input) -> {}(Hidden) -> {}(Output)", n_input, n_hidden, n_output);
    info!("Activation Functions: {} (Hidden), {} (Output)", hidden_activation.name(), output_activation.name());

    let mlp = MLP::new(n_input, n_hidden, n_output, hidden_activation, output_activation, loss_function);
    info!("MLP model created successfully.");
    Ok(mlp)
}

impl MLP {
    /// 단일 샘플 x에 대해 순전파 수행 (자동미분 사용)
    ///
    /// x: Variable wrapping Tensor of shape [input_node, 1]
    /// 반환: (z, y) - 모두 Variable로 래핑됨
    ///   - z: 은닉층 활성화값( bias row 포함 ) → shape = [(hidden_node+1), 1]
    ///   - y: 출력층 활성화값 → shape = [output_node, 1]
    pub fn apply(&self, x: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 은닉층: u_h = W1 * x + b1
        let uh_pre = matmul.apply(&[&self.w1, x])?;
        let uh = add.apply(&[&uh_pre, &self.b1])?;

        // 2) 은닉층 활성화: a_h = activation(u_h)
        let ah = self.hidden_activation.apply(&[&uh])?;

        // 3) 출력층: u_o = W2 * a_h + b2
        let uo_pre = matmul.apply(&[&self.w2, &ah])?;
        let uo = add.apply(&[&uo_pre, &self.b2])?;

        // 4) 출력층 활성화: y = activation(u_o)
        // 다중 클래스 분류에는 Softmax가 표준입니다.
        let y = self.output_activation.apply_with_label(&[&uo], "output")?;

        Ok(y)
    }

    pub fn forward(&self, x: &Tensor) -> MlResult<Tensor> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 은닉층: u_h = W1 * x + b1
        let uh_pre = matmul.forward(&[&self.w1.tensor(), x])?.remove(0);
        let uh = add.forward(&[&uh_pre, &self.b1.tensor()])?.remove(0);

        // 2) 은닉층 활성화: a_h = activation(u_h)
        let ah = self.hidden_activation.forward(&[&uh])?.remove(0);

        // 3) 출력층: u_o = W2 * a_h + b2
        let uo_pre = matmul.forward(&[&self.w2.tensor(), &ah])?.remove(0);
        let uo = add.forward(&[&uo_pre, &self.b2.tensor()])?.remove(0);

        // 4) 출력층 활성화: y = activation(u_o)
        // 다중 클래스 분류에는 Softmax가 표준입니다.
        let y = self.output_activation.forward(&[&uo])?.remove(0);

        Ok(y)
    }

    pub fn train(
        &mut self,
        x_set: &[Arc<Variable>],
        t_set: &[Arc<Variable>],
        learning_rate: f32,
        epochs: usize,
        tolerance: f32,
    ) -> MlResult<()> {
        let n_batches = x_set.len();
        let training_start_time = Instant::now();
        let lr = scalar!(learning_rate);

        // 에포크 진행 상태를 보여주는 프로그레스 바 설정
        let epoch_bar = ProgressBar::new(epochs as u64);
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Epochs ({eta})")
                .unwrap()
                .progress_chars("█▉ "),
        );

        info!("Initial error calculation...");
        let mut epoch_start_time = Instant::now();
        let mut last_loss = self.compute_total_error(x_set, t_set, &self.loss_function)?;
        let epoch_duration = epoch_start_time.elapsed();
        info!(
                "Epoch {:>3}/{:<3} | initial loss: {:.6} | Duration: {:.2?}",
                0.000,
                epochs,
                last_loss,
                epoch_duration
            );

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            epoch_start_time = Instant::now();

            // 각 배치에 대한 학습 수행
            for (x, t) in x_set.iter().zip(t_set.iter()) {
                // 순전파 및 손실 계산
                ComputationGraph::reset_graph();
                let y = self.apply(x)?;
                let loss_var = self.loss_function.apply_with_label(&[&y, &t], "loss")?;
                
                total_loss += loss_var.tensor().data()[0];
                loss_var.backward()?;

                if self.w1.grad().unwrap().data()[0].is_nan() || self.w2.grad().unwrap().data()[0].is_infinite() || self.b1.grad().unwrap().data()[0].is_nan() || self.b2.grad().unwrap().data()[0].is_infinite() {
                    error!("gradient is NaN or infinity: {}. Suspended training.", total_loss);
                    return Err(MlError::StringError("During training, numerical instability occurs".to_string()));
                }
                
                self.update(&lr)?;
                self.zero_grad()?;
            }

            let avg_loss = total_loss / n_batches as f32;
            let epoch_duration = epoch_start_time.elapsed();

            // 에포크 진행 바 업데이트
            epoch_bar.inc(1);

            // 콘솔과 파일에 에포크 요약 정보 로깅
            info!(
                "Epoch {:>3}/{:<3} | Avg Loss: {:.6} | Loss Chg: {:+.6} | Duration: {:.2?}",
                epoch + 1,
                epochs,
                avg_loss,
                avg_loss - last_loss,
                epoch_duration
            );

            // 수렴 조건 확인 (조기 종료)
            if (last_loss - avg_loss).abs() < tolerance {
                info!("✅ Loss has converged at epoch {}. Early stopping.", epoch + 1);
                // 남은 단계를 모두 채워서 프로그레스 바를 100%로 만듭니다.
                epoch_bar.finish_with_message("Converged");
                break;
            }
            last_loss = avg_loss;

            // 마지막 에포크 완료 시
            if epoch == epochs - 1 {
                epoch_bar.finish_with_message("Completed");
            }
        }

        let total_duration = training_start_time.elapsed();
        info!("🏁 Total training time: {:.2?}. Final average loss: {:.6}", total_duration, last_loss);

        Ok(())
    }

    pub fn compute_total_error(&self, X: &[Arc<Variable>], T: &[Arc<Variable>], loss_function: &GlobalFunction) -> MlResult<f32> {
        let mut total_loss = 0.0;
        for m in 0..X.len() {
            let y = var_input!(self.forward(&X[m].tensor())?);
            let loss = loss_function.forward(&[&y.tensor(), &T[m].tensor()])?.remove(0);
            total_loss += loss.data()[0];
        }
        Ok(total_loss / X.len() as f32)
    }

    pub fn update(&mut self, lr: &Tensor) -> MlResult<()> {
        self.w1.sub_tensor(self.w1.grad().unwrap() * lr)?;
        self.w2.sub_tensor(self.w2.grad().unwrap() * lr)?;
        // self.b1.sub_tensor(self.b1.grad().unwrap() * lr)?;
        // self.b2.sub_tensor(self.b2.grad().unwrap() * lr)?;
        Ok(())
    }


    pub fn zero_grad(&mut self) -> MlResult<()> {
        self.w1.clear_grad();
        self.w2.clear_grad();
        // self.b1.clear_grad();
        // self.b2.clear_grad();
        Ok(())
    }
}

pub fn train_model(
    mlp: &mut MLP,
    x_train: &[Arc<Variable>],
    t_train: &[Arc<Variable>],
    learning_rate: f32,
    epochs: usize,
    tolerance: f32,
) -> MlResult<()> {
    info!("Starting model training...");
    info!("Training Parameters: LR={}, Max Epochs={}, Tolerance={}", learning_rate, epochs, tolerance);

    mlp.train(x_train, t_train, learning_rate, epochs, tolerance)?;

    info!("Model training finished.");
    Ok(())
}