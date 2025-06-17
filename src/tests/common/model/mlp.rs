use tracing::error;
use super::*;

impl Model for MLP {
    /// n_input : 입력 뉴런 개수
    /// n_hidden: 은닉 뉴런 개수
    /// n_output: 출력 뉴런 개수
    /// hidden_activation: 은닉층에 적용할 활성화 함수
    /// output_activation: 출력층에 적용할 활성화 함수
    fn new(
        layer_parms: &[usize],
        activations: &[GlobalFunction],
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

        Self { w1, w2, b1, b2, hidden_activation: activations[0].clone(), output_activation: activations[1].clone(), loss_function }
    }

    #[cfg(feature = "enableBackpropagation")]
    fn train(&mut self, x_set: &[Arc<Variable>], t_set: &[Arc<Variable>], epochs: usize, learning_rate: f32, tolerance: f32) -> MlResult<()> {
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
        let mut last_loss = self.compute_total_error(&x_set, &t_set, &self.loss_function)?;
        let epoch_duration = epoch_start_time.elapsed();
        info!(
                "Epoch {:>3}/{:<3} | initial loss: {:.6} | Duration: {:.2?}",
                0,
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

    #[cfg(feature = "enableBackpropagation")]
    fn apply(&self, x: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 첫 번째 은닉층
        let uh1_pre = matmul.apply(&[&self.w1, x])?;
        let uh1 = add.apply(&[&uh1_pre, &self.b1])?;
        let ah1 = self.hidden_activation.apply(&[&uh1])?;

        // 2) 두 번째 은닉층: 첫 번째 은닉층의 출력(ah1)을 입력으로 받음
        let uh2_pre = matmul.apply(&[&self.w2, &ah1])?;
        add.apply(&[&uh2_pre, &self.b2])
    }

    fn predict(&self, x: &Tensor) -> MlResult<Tensor> {
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 1) 은닉층: u_h = W1 * x + b1
        let uh1_pre = matmul.forward(&[&self.w1.tensor(), x])?.remove(0);
        let uh1 = add.forward(&[&uh1_pre, &self.b1.tensor()])?.remove(0);
        let ah1 = self.hidden_activation.forward(&[&uh1])?.remove(0);

        // 3) 은닉층: u_h = W1 * x + b1
        let uh2_pre = matmul.forward(&[&self.w2.tensor(), &ah1])?.remove(0);
        let uh2 = add.forward(&[&uh2_pre, &self.b2.tensor()])?.remove(0);
        let ah2 = self.output_activation.forward(&[&uh2])?.remove(0);

        Ok(ah2)
    }

    #[cfg(feature = "enableBackpropagation")]
    fn update(&mut self, lr: &Tensor) -> MlResult<()> {
        self.w1.sub_tensor(self.w1.grad().unwrap() * lr)?;
        self.w2.sub_tensor(self.w2.grad().unwrap() * lr)?;
        self.b1.sub_tensor(self.b1.grad().unwrap() * lr)?;
        self.b2.sub_tensor(self.b2.grad().unwrap() * lr)?;
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
}