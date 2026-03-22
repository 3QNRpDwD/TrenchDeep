use crate::nn::Layer;
use super::*;

impl SoftmaxRegression {
    pub fn new(
        layer_parms: &[usize],
        activation: GlobalFunction,
        loss_function: GlobalFunction,
    ) -> Self {
        let n_input = layer_parms[0];
        let n_output = layer_parms[1];
        // He 초기화 또는 Xavier 초기화와 같은 더 나은 가중치 초기화 방법을 고려할 수 있음
        let w1_data: Vec<f32> = (0..n_output * n_input)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5) // 0을 중심으로 분포
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_output, n_input]).unwrap(),
            "weight_1"
        );

        // bias 항들 초기화
        let b1_data: Vec<f32> = vec![0.0; n_output]; // 0으로 초기화하는 것이 일반적
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_output, 1]).unwrap(),
            "bias_1"
        );

        Self { w1, b1, activation, loss_function }
    }
}


impl Model for SoftmaxRegression {
    #[cfg(feature = "enableBackward")]
    fn train(&mut self, x_set: &[&Variable], t_set: &[&Variable], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()> {
        // [FIX 4] n_batches → n_samples: 실제로 샘플 1개씩 처리하는 SGD 구조이므로 명칭 수정
        let n_samples = x_set.len();
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

        // [FIX 2] 초기 손실은 로그 표시용으로만 사용.
        // compute_total_error의 내부 계산 방식(합산/평균/정규화 여부)이
        // 훈련 루프의 avg_loss 계산 방식과 다를 수 있으므로,
        // 수렴 판정용 last_loss는 f32::INFINITY로 초기화하여
        // 첫 에폭의 avg_loss와 잘못 비교되는 것을 방지합니다.
        let initial_display_loss = self.compute_total_error(&x_set, &t_set)?;
        let mut last_loss = f32::INFINITY; // 수렴 판정 전용
        let epoch_duration = epoch_start_time.elapsed();
        let initial_log = format!("Initial loss: {:.6} | Avg Acc: {:>6.2}% | Duration: {:.2?}", initial_display_loss, 0, epoch_duration);
        epoch_bar.set_message(initial_log.clone());

        // [FIX 2] NaN/Inf 검사용 클로저: 슬라이스 전체 원소를 검사하고 Inf도 포함
        let has_invalid = |data: &[f32]| data.iter().any(|x| x.is_nan() || x.is_infinite());

        for epoch in 0..epochs {
            let mut total_correct = 0usize;
            // [FIX 5] 손실 평균 분모를 위한 별도 카운터.
            // total_samples는 argmax가 유효한 샘플만 세므로, 손실 분모로 쓰면
            // 정확도와 손실의 분모가 불일치합니다. 별도 카운터로 분리합니다.
            let mut total_samples = 0usize;     // 정확도 분모: argmax 유효 샘플 수
            let mut total_loss_count = 0usize;  // 손실 분모: 실제 처리된 전체 샘플 수
            let mut total_loss = 0.0f32;
            epoch_start_time = Instant::now();

            let batch_bar = multi_bar.add(ProgressBar::new(n_samples as u64));
            let formatted_template = format!(
                "  > Epoch {:>3}/{:<3} [ {{wide_bar:.green/blue}} ] {{pos}}/{{len}} Batches ({{eta}}) | {{msg}}",
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
                let forward_start = Instant::now();

                let y = self.apply(x)?;
                let loss_var = self.loss_function.apply_with_label(&[&y, &t], "loss")?;

                let forward_duration = forward_start.elapsed();
                let y_pred_idx = utils::argmax(y.tensor().data());
                let t_true_idx = utils::argmax(t.tensor().data());

                if let (Some(pred_idx), Some(true_idx)) = (y_pred_idx, t_true_idx) {
                    if pred_idx == true_idx {
                        total_correct += 1;
                    }
                    total_samples += 1; // argmax 유효 샘플만 정확도 분모에 포함
                }

                total_loss += loss_var.tensor().data()[0];
                total_loss_count += 1; // 손실은 모든 처리 샘플을 분모로 사용

                let backward_start = Instant::now();
                loss_var.backward()?;
                let backward_duration = backward_start.elapsed();

                // [FIX 2] NaN/Inf 검사: [0]번 원소만이 아닌 w1, b1 전체 슬라이스 검사
                // Gradient explosion 시 NaN보다 Inf가 먼저 발생하는 경우가 많으므로
                // is_infinite()도 함께 검사합니다.
                if has_invalid(self.w1.grad().data()) || has_invalid(self.b1.grad().data()) {
                    epoch_bar.abandon_with_message("❌ Error: NaN/Inf Gradient");
                    batch_bar.abandon_with_message("NaN/Inf Gradient");
                    error!("gradient is NaN or Infinite: {}. Suspended training.", total_loss);
                    return Err(MlError::StringError("During training, numerical instability occurs".to_string()));
                }

                self.update(&lr)?;

                let grad_sq_sum = self.w1.grad().data().iter().map(|&g| g * g).sum::<f32>()
                    + self.b1.grad().data().iter().map(|&g| g * g).sum::<f32>();
                let grad_norm = grad_sq_sum.sqrt();

                let update_sq_sum = self.w1.grad().data().iter().map(|&g| (learning_rate * g).powi(2)).sum::<f32>()
                    + self.b1.grad().data().iter().map(|&g| (learning_rate * g).powi(2)).sum::<f32>();
                let update_norm = update_sq_sum.sqrt();

                let weight_sq_sum = self.w1.tensor().data().iter().map(|&w| w * w).sum::<f32>()
                    + self.b1.tensor().data().iter().map(|&w| w * w).sum::<f32>();
                let weight_norm = weight_sq_sum.sqrt();

                let update_ratio = if weight_norm > 1e-6 { update_norm / weight_norm } else { 0.0 };

                self.zero_grad()?;

                let batch_log_message = format!(
                    "FW: {:>7.2?} | BW: {:>7.2?} | GN: {:.2e}| UR: {:.2e}",
                    forward_duration,
                    backward_duration,
                    grad_norm,
                    update_ratio
                );
                batch_bar.set_message(batch_log_message);
                batch_bar.inc(1);
            }

            batch_bar.finish_and_clear();

            // [FIX 5] avg_loss 분모를 n_samples(전체)가 아닌 total_loss_count(실제 처리)로 계산
            let avg_loss = if total_loss_count > 0 { total_loss / total_loss_count as f32 } else { 0.0 };
            let epoch_duration = epoch_start_time.elapsed();
            let epoch_accuracy = if total_samples > 0 {
                (total_correct as f32 / total_samples as f32) * 100.0
            } else {
                0.0
            };
            // let epoch_accuracy = evaluate_model(self, &x_set, &t_set)?;

            let log_message = format!(
                "AL: {:.6} | LC: {:+.6} | AC: {:>6.2}% | Duration: {:.2?}",
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

    #[cfg(feature = "enableBackward")]
    fn apply(&mut self, x: &Variable) -> MlResult<Variable> {
        let mut matmul = Matmul::new()?;

        let uh1_pre = matmul.apply(&[&self.w1, x])?;
        Ok(&uh1_pre + &self.b1)
    }

    fn predict(&mut self, x: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 은닉층: u_h = W1 * x + b1
        let uh1_pre = matmul.forward(&[self.w1.tensor(), x])?.remove(0);
        let uh1 = add.forward(&[&uh1_pre, self.b1.tensor()])?.remove(0);
        let ah1 = self.activation.forward(&[&uh1])?.remove(0);

        Ok(ah1)
    }

    #[cfg(feature = "enableBackward")]
    fn update(&mut self, lr: &dyn TensorBase) -> MlResult<()> {
        self.w1 -= self.w1.grad() as &dyn TensorBase * lr;
        self.b1 -= self.b1.grad() as &dyn TensorBase * lr;
        Ok(())
    }

    #[cfg(feature = "enableBackward")]
    fn zero_grad(&mut self) -> MlResult<()> {
        self.w1.clear_grad();
        self.b1.clear_grad();
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
            let logit_tensor = {
                let matmul = Matmul::new()?;
                let  add = Add::new()?;
                let uh1_pre = matmul.forward(&[self.w1.tensor(), X[m].tensor()])?.remove(0);
                add.forward(&[&uh1_pre, self.b1.tensor()])?.remove(0)
            };

            // logit을 사용해 손실을 계산합니다.
            let loss = self.loss_function.forward(&[&logit_tensor, T[m].tensor()])?.remove(0);
            total_loss += loss.data()[0];
        }
        Ok(total_loss / X.len() as f32)
    }   
}