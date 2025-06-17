use tracing::error;
use super::*;

impl Model for SoftmaxRegression {
    fn new(
        layer_parms: &[usize] ,
        activations: &[GlobalFunction],
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

        Self { w1, b1, output_activation: activations[0].clone(), loss_function }
    }

    #[cfg(feature = "enableBackpropagation")]
    fn train(&mut self, x_set: &[Arc<Variable>], t_set: &[Arc<Variable>], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()> {
        let n_batches = x_set.len();
        let training_start_time = Instant::now();
        let lr = scalar!(learning_rate);

        // --- 1. MultiProgress 객체 생성 ---
        let multi_bar = MultiProgress::new();

        // --- 2. 에포크 프로그레스 바 설정 (ETA 및 통합 로그 추가) ---
        let epoch_bar = multi_bar.add(ProgressBar::new(epochs as u64));
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [ {wide_bar:.cyan/blue} ] {pos}/{len} Epochs ({eta}) | {msg}")
                .unwrap()
                .progress_chars("▉ "),
        );

        info!("Initial error calculation...");
        let mut epoch_start_time = Instant::now();
        let mut last_loss = self.compute_total_error(x_set, t_set, &self.loss_function)?;
        let epoch_duration = epoch_start_time.elapsed();
        let initial_log = format!("Initial loss: {:.6} | Duration: {:.2?}", last_loss, epoch_duration);
        epoch_bar.set_message(initial_log.clone());
    //     info!(
    //     "Epoch {:>3}/{:<3} | {}",
    //     0,
    //     epochs,
    //     initial_log
    // );


        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            epoch_start_time = Instant::now();

            // --- 3. 배치 프로그레스 바 설정 (기존과 동일) ---
            let batch_bar = multi_bar.add(ProgressBar::new(n_batches as u64));
            let formatted_template = format!(
                "  > Epoch {:>3}/{:<3} [ {{wide_bar:.blue/green}} ] {{pos}}/{{len}} Batches ({{eta}})",
                epoch + 1,
                epochs
            );
            batch_bar.set_style(
                ProgressStyle::default_bar()
                    .template(&formatted_template)
                    .unwrap()
                    .progress_chars("█ "),
            );


            for (x, t) in x_set.iter().zip(t_set.iter()) {
                ComputationGraph::reset_graph();
                let y = self.apply(x)?;
                let loss_var = self.loss_function.apply_with_label(&[&y, &t], "loss")?;

                total_loss += loss_var.tensor().data()[0];
                loss_var.backward()?;

                if self.w1.grad().unwrap().data()[0].is_nan() || self.b1.grad().unwrap().data()[0].is_nan() {
                    epoch_bar.abandon_with_message("❌ Error: NaN Gradient");
                    batch_bar.abandon_with_message("NaN Gradient");
                    error!("gradient is NaN or infinity: {}. Suspended training.", total_loss);
                    return Err(MlError::StringError("During training, numerical instability occurs".to_string()));
                }

                self.update(&lr)?;
                self.zero_grad()?;
                batch_bar.inc(1);
            }

            batch_bar.finish_and_clear();

            let avg_loss = total_loss / n_batches as f32;
            let epoch_duration = epoch_start_time.elapsed();

            // --- 4. 에포크 프로그레스 바 메시지 업데이트 및 진행 ---
            let log_message = format!(
                "Avg Loss: {:.6} | Loss Chg: {:+.6} | Duration: {:.2?}",
                avg_loss,
                avg_loss - last_loss,
                epoch_duration
            );
            epoch_bar.set_message(log_message); // 메시지 업데이트
            epoch_bar.inc(1); // 에포크 진행

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
    fn apply(&self, x: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 첫 번째 은닉층
        let uh1_pre = matmul.apply(&[&self.w1, x])?;
        add.apply(&[&uh1_pre, &self.b1])
    }

    fn predict(&self, x: &Tensor) -> MlResult<Tensor> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 은닉층: u_h = W1 * x + b1
        let uh1_pre = matmul.forward(&[&self.w1.tensor(), x])?.remove(0);
        let uh1 = add.forward(&[&uh1_pre, &self.b1.tensor()])?.remove(0);
        let ah1 = self.output_activation.forward(&[&uh1])?.remove(0);

        Ok(ah1)
    }

    #[cfg(feature = "enableBackpropagation")]
    fn update(&mut self, lr: &Tensor) -> MlResult<()> {
        self.w1.sub_tensor(self.w1.grad().unwrap() * lr)?;
        self.b1.sub_tensor(self.b1.grad().unwrap() * lr)?;
        Ok(())
    }

    #[cfg(feature = "enableBackpropagation")]
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
}