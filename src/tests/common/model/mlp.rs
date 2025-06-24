use tracing::error;
use crate::nn::Sequential;
use super::*;

impl MLP {
    pub fn new(
        layer_parms: &[usize],
        layer: Sequential,
        loss_function: GlobalFunction,
    ) -> Self {
        let n_input = layer_parms[0];
        let n_hidden = layer_parms[1];
        let n_output = layer_parms[2];
        // He 초기화 또는 Xavier 초기화와 같은 더 나은 가중치 초기화 방법을 고려할 수 있음
        let w1_data: Vec<f32> = (0..n_hidden * n_input)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5) // 0을 중심으로 분포
            .collect();
        let w1 = var_with_label!(
                Tensor::from_vec(w1_data, &[n_hidden, n_input]).unwrap(),
                "weight_1"
            );

        let w2_data: Vec<f32> = (0..n_output * n_hidden)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();
        let w2 = var_with_label!(
                Tensor::from_vec(w2_data, &[n_output, n_hidden]).unwrap(),
                "weight_2"
            );

        // bias 항들 초기화
        let b1_data: Vec<f32> = vec![0.0; n_hidden]; // 0으로 초기화하는 것이 일반적
        let b1 = var_with_label!(
                Tensor::from_vec(b1_data, &[n_hidden, 1]).unwrap(),
                "bias_1"
            );

        let b2_data: Vec<f32> = vec![0.0; n_output];
        let b2 = var_with_label!(
                Tensor::from_vec(b2_data, &[n_output, 1]).unwrap(),
                "bias_2"
            );

        Self { w1, w2, b1, b2, layer , loss_function }
    }
}

impl Model for MLP {

    #[cfg(feature = "enableBackpropagation")]
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()> {
        let n_batches = x_set.len();
        let training_start_time = Instant::now();
        let lr = Tensor::scalar(learning_rate);
        let multi_bar = MultiProgress::new();
        let epoch_bar = multi_bar.add(ProgressBar::new(epochs as u64));
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [ {wide_bar:.cyan/blue} ] {pos}/{len} Epochs ({eta}) | {msg}")
                .unwrap()
                .progress_chars("▉ "),
        );

        info!("Initial error calculation...");
        let mut epoch_start_time = Instant::now();
        let mut last_loss = self.compute_total_error(&x_set, &t_set)?;
        let epoch_duration = epoch_start_time.elapsed();
        let initial_log = format!("Initial loss: {:.6} | Avg Acc: {:>6.2}% | Duration: {:.2?}", last_loss, 0, epoch_duration);
        epoch_bar.set_message(initial_log.clone());

        for epoch in 0..epochs {
            let mut total_correct = 0;
            let mut total_samples = 0;
            let mut total_loss = 0.0;
            epoch_start_time = Instant::now();

            // --- 1. 배치 프로그레스 바 설정 (템플릿 수정) ---
            let batch_bar = multi_bar.add(ProgressBar::new(n_batches as u64));
            let formatted_template = format!(
                // {msg} 플레이스홀더를 추가하여 순전파/역전파 시간 정보를 표시할 공간을 만듭니다.
                "  > Epoch {:>3}/{:<3} [ {{wide_bar:.blue/green}} ] {{pos}}/{{len}} Batches ({{eta}}) | {{msg}}",
                epoch + 1,
                epochs
            );
            batch_bar.set_style(
                ProgressStyle::default_bar()
                    .template(&formatted_template)
                    .unwrap()
                    .progress_chars("█ "),
            );

            let mut rng = rng();
            let mut combined_train_data: Vec<_> = x_set.into_iter().zip(t_set.into_iter()).collect();
            combined_train_data.shuffle(&mut rng);

            for (x, t) in combined_train_data.into_iter() {
                ComputationGraph::reset_graph();

                // --- 2. 순전파 시간 측정 ---
                let forward_start = Instant::now();
                let y = self.apply(x)?;
                let loss_var = self.loss_function.apply_with_label(&[&y, &t], "loss")?;
                let forward_duration = forward_start.elapsed();
                let y_pred_idx = utils::argmax(y.tensor().data()); // 예측값의 argmax
                let t_true_idx = utils::argmax(t.tensor().data()); // 실제 정답의 argmax
                if let (Some(pred_idx), Some(true_idx)) = (y_pred_idx, t_true_idx) {
                    if pred_idx == true_idx {
                        total_correct += 1;
                    }
                    total_samples += 1;
                }
                total_loss += loss_var.tensor().data()[0];

                // --- 3. 역전파 시간 측정 ---
                let backward_start = Instant::now();
                loss_var.backward()?;
                let backward_duration = backward_start.elapsed();
                let grad_norm = self.w1.grad().data().iter().map(|&x| x * x).sum::<f32>().sqrt();

                if self.w1.grad().data()[0].is_nan() || self.w2.grad().data()[0].is_nan() || self.b1.grad().data()[0].is_nan() || self.b2.grad().data()[0].is_nan() {
                    epoch_bar.abandon_with_message("❌ Error: NaN Gradient");
                    batch_bar.abandon_with_message("NaN Gradient");
                    error!("gradient is NaN or infinity: {}. Suspended training.", total_loss);
                    return Err(MlError::StringError("During training, numerical instability occurs".to_string()));
                }

                self.update(&lr)?;
                let update_norm = self.w1.grad().data().iter().map(|&g| (learning_rate * g).powi(2)).sum::<f32>().sqrt();
                let weight_norm = self.w1.tensor().data().iter().map(|&w| w * w).sum::<f32>().sqrt();
                let update_ratio = if weight_norm > 1e-6 { update_norm / weight_norm } else { 0.0 };

                self.zero_grad()?;

                // ... batch_log_message 포맷팅 수정
                let batch_log_message = format!(
                    "Forward: {:>7.2?} | Backward: {:>7.2?} | Grad Norm: {:.2e}| Update Ratio: {:.2e}", // 과학적 표기법(e) 사용
                    forward_duration,
                    backward_duration,
                    grad_norm,
                    update_ratio
                );
                batch_bar.set_message(batch_log_message);
                batch_bar.inc(1);
            }

            batch_bar.finish_and_clear();

            let avg_loss = total_loss / n_batches as f32;
            let epoch_duration = epoch_start_time.elapsed();
            let epoch_accuracy = if total_samples > 0 {
                (total_correct as f32 / total_samples as f32) * 100.0
            } else {
                0.0
            };

            let log_message = format!(
                "Avg Loss: {:.6} | Loss Chg: {:+.6} | Avg Acc: {:>6.2}% | Duration: {:.2?}",
                avg_loss,
                avg_loss - last_loss,
                epoch_accuracy,
                epoch_duration
            );
            epoch_bar.set_message(log_message);
            epoch_bar.inc(1);

            if (last_loss - avg_loss).abs() < tolerance {
                epoch_bar.finish_with_message("✅ Converged");
                info!("Loss has converged. Early stopping.");
                break;
            }
            last_loss = avg_loss;

            if epoch == epochs - 1 {
                epoch_bar.finish_with_message("✅ Completed");
            }
        }

        let total_duration = training_start_time.elapsed();
        info!("🏁 Total training time: {:.2?}. Final average loss: {:.6}", total_duration, last_loss);

        Ok(())
    }

    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;
        let mut add = Add::new()?;

        let uh1_pre = matmul.apply(&[&self.w1, x])?;
        let uh1 = add.apply(&[&uh1_pre, &self.b1])?;
        let ah1 = self.layer.apply(&uh1)?;

        let uh2_pre = matmul.apply(&[&self.w2, &ah1])?;
        let uh2 = add.apply(&[&uh2_pre, &self.b2])?;
        let ah2 = self.layer.apply(&uh2)?;

        Ok(ah2)
    }

    fn predict(&mut self, x: &Tensor) -> MlResult<GlobalTensor<f32>> {
        let mut matmul = Matmul::new()?;
        let mut add = Add::new()?;

        let uh1_pre = matmul.forward(&[self.w1.tensor(), x])?.remove(0);
        let uh1 = add.forward(&[&uh1_pre, self.b1.tensor()])?.remove(0);
        let ah1 = self.layer.predict(&uh1)?;

        let uh2_pre = matmul.forward(&[self.w2.tensor(), &ah1])?.remove(0);
        let uh2 = add.forward(&[&uh2_pre, self.b2.tensor()])?.remove(0);
        let ah2 = self.layer.predict(&uh2)?;

        Ok(ah2)
    }

    #[cfg(feature = "enableBackpropagation")]
    fn update(&self, lr: &dyn TensorBase) -> MlResult<()> {
        self.w1.sub_tensor(self.w1.grad() as &dyn TensorBase * lr)?;
        self.w2.sub_tensor(self.w2.grad() as &dyn TensorBase * lr)?;
        self.b1.sub_tensor(self.b1.grad() as &dyn TensorBase * lr)?;
        self.b2.sub_tensor(self.b2.grad() as &dyn TensorBase * lr)?;
        Ok(())
    }

    #[cfg(feature = "enableBackpropagation")]
    fn zero_grad(&mut self) -> MlResult<()> {
        self.w1.clear_grad();
        self.w2.clear_grad();
        self.b1.clear_grad();
        self.b2.clear_grad();
        Ok(())
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

    fn compute_total_error(&mut self, X: &[&Variable], T: &[&Variable]) -> MlResult<f32> {
        let mut total_loss = 0.0;
        for m in 0..X.len() {
            let y = self.predict(X[m].tensor())?;
            let loss = self.loss_function.forward(&[&y, T[m].tensor()])?.remove(0);
            total_loss += loss.data()[0];
        }
        Ok(total_loss / X.len() as f32)
    }
}