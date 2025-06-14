
use crate::tensor::{OPERATOR_STORAGE, Tensor, TensorBase, AutogradFunction};
use crate::nn::activation::Sigmoid;
use crate::tensor::Variable;
use std::fmt;
use std::sync::Arc;
use crate::tensor::operators::{Add, Function, Matmul, Mul, Square, Sub, Sum};
use crate::{MlError, MlResult, scalar, var_with_label};
use crate::loss::MeanSquaredError;
use log::{info, debug, warn, error, trace};

pub struct MLP {
    pub w1: Arc<Variable>, // shape = [hidden_node, input_node + 1]
    pub w2: Arc<Variable>, // shape = [output_node, hidden_node + 1]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    pub b2: Arc<Variable>, // shape = [output_node, 1]
}

impl fmt::Debug for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  w1.shape = {:?}, w2.shape = {:?} ",
                 self.w1.tensor().shape(),
                 self.w2.tensor().shape())?;
        writeln!(f, "}}")
    }
}


impl MLP {
    /// n_input : 입력 뉴런 개수
    /// n_hidden: 은닉 뉴런 개수
    /// n_output: 출력 뉴런 개수
    pub fn new(n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        let w1_data: Vec<f32> = (0..n_hidden * n_input)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_hidden, n_input]).unwrap(),
            "weight_1"
        );

        let w2_data: Vec<f32> = (0..n_output * n_hidden)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let w2 = var_with_label!(
            Tensor::from_vec(w2_data, &[n_output, n_hidden]).unwrap(),
            "weight_2"
        );

        // bias 항들 초기화
        let b1_data: Vec<f32> = (0..n_hidden)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_hidden, 1]).unwrap(),
            "bias_1"
        );

        let b2_data: Vec<f32> = (0..n_output)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let b2 = var_with_label!(
            Tensor::from_vec(b2_data, &[n_output, 1]).unwrap(),
            "bias_2"
        );
        Self { w1, w2, b1, b2 }
    }

    /// 단일 샘플 x에 대해 순전파 수행 (자동미분 사용)
    ///
    /// x: Variable wrapping Tensor of shape [input_node, 1]
    /// 반환: (z, y) - 모두 Variable로 래핑됨
    ///   - z: 은닉층 활성화값( bias row 포함 ) → shape = [(hidden_node+1), 1]
    ///   - y: 출력층 활성화값 → shape = [output_node, 1]
    pub fn forward(&self, x: &Arc<Variable>) -> MlResult<(Arc<Variable>, Arc<Variable>)> {
        let sigmoid = Sigmoid::new()?;
        let matmul = Matmul::new()?;
        let add = Add::new()?; // Add 연산 추가

        // 1) 은닉층: u_h = W1 * x + b1
        let uh_pre = matmul.apply(&[&self.w1, x])?;
        let uh = add.apply(&[&uh_pre, &self.b1])?;

        // 2) 은닉층 활성화: a_h = sigmoid(u_h)
        let ah = sigmoid.apply(&[&uh])?;

        // 3) 출력층: u_o = W2 * a_h + b2
        let uo_pre = matmul.apply(&[&self.w2, &ah])?;
        let uo = add.apply(&[&uo_pre, &self.b2])?;

        // 4) 출력층 활성화: y = sigmoid(u_o)
        let y = sigmoid.apply_with_label(&[&uo], "output")?;
        Ok((ah, y)) // 은닉층 출력과 최종 출력 반환
    }

    #[cfg(feature = "enableBackpropagation")]
    pub fn train(
        &mut self,
        X: &Vec<Arc<Variable>>,
        T: &Vec<Arc<Variable>>,
        eta: f32,
        max_iter: usize,
        tol: f32,
    ) -> MlResult<()> {
        info!("=== 모델 훈련 시작 ===");
        info!("훈련 설정: 샘플수={}, 학습률={}, 최대반복={}, 허용오차={}",
         X.len(), eta, max_iter, tol);

        let start_time = std::time::Instant::now();

        // 연산자 초기화
        debug!("손실 함수 및 연산자 초기화 중...");
        let sub = Sub::new().map_err(|e| {
            error!("Sub 연산자 초기화 실패: {:?}", e);
            e
        })?;
        let square = Square::new().map_err(|e| {
            error!("Square 연산자 초기화 실패: {:?}", e);
            e
        })?;
        let sum = Sum::new().map_err(|e| {
            error!("Sum 연산자 초기화 실패: {:?}", e);
            e
        })?;
        let mse = MeanSquaredError::new().map_err(|e| {
            error!("MeanSquaredError 연산자 초기화 실패: {:?}", e);
            e
        })?;

        let n_samples = X.len();
        let mut resid = tol * 2.0;
        let mut iter = 1;
        let lr = scalar!(eta);

        debug!("연산자 초기화 완료");

        // 초기 오차 계산
        info!("초기 오차 계산 중...");
        let mut e_prev = self.compute_error(X, T).map_err(|e| {
            error!("초기 오차 계산 실패: {:?}", e);
            e
        })?;

        info!("에포크 {}: 오차 = {:.6}", iter - 1, e_prev);
        debug!("초기 잔차(residual): {:.6}, 허용 오차: {:.6}", resid, tol);

        let mut best_error = e_prev;
        let mut best_epoch = 0;
        let mut no_improvement_count = 0;
        const PATIENCE: usize = 10; // 조기 종료를 위한 patience

        info!("훈련 루프 시작...");

        while resid >= tol && iter <= max_iter {
            let epoch_start = std::time::Instant::now();
            debug!("에포크 {} 시작 (잔차: {:.6})", iter, resid);

            // 1 epoch 동안 샘플별로 순전파→역전파→업데이트
            let mut sample_errors = Vec::new();

            for m in 0..n_samples {
                trace!("샘플 {}/{} 처리 중", m + 1, n_samples);

                let x_m = &X[m];
                let t_m = &T[m];

                // === 순전파 (자동미분 그래프 구성) ===
                let (_z, y) = self.forward(x_m).map_err(|e| {
                    error!("샘플 {} 순전파 실패: {:?}", m, e);
                    e
                })?;

                trace!("샘플 {} 순전파 완료", m);

                #[cfg(feature = "enableBackpropagation")]
                {
                    // === 손실 함수 계산 ===
                    let loss_val = mse.apply_with_label(&[&y, t_m], "loss").map_err(|e| {
                        error!("샘플 {} 차이 계산 실패: {:?}", m, e);
                        e
                    })?;

                    // === 역전파 (자동미분) ===
                    trace!("샘플 {} 역전파 시작", m);
                    loss_val.backward().map_err(|e| {
                        error!("샘플 {} 역전파 실패: {:?}", m, e);
                        e
                    })?;

                    // 가중치 업데이트 전 그래디언트 체크
                    if let Some(w1_grad) = self.w1.grad() {
                        let w1_grad_norm = w1_grad.data().iter().map(|x| x * x).sum::<f32>().sqrt();
                        trace!("W1 그래디언트 노름: {:.6}", w1_grad_norm);

                        if w1_grad_norm > 10.0 {
                            warn!("W1 그래디언트가 큽니다: {:.6} (그래디언트 폭발 가능성)", w1_grad_norm);
                        }
                    }

                    if let Some(w2_grad) = self.w2.grad() {
                        let w2_grad_norm = w2_grad.data().iter().map(|x| x * x).sum::<f32>().sqrt();
                        trace!("W2 그래디언트 노름: {:.6}", w2_grad_norm);

                        if w2_grad_norm > 10.0 {
                            warn!("W2 그래디언트가 큽니다: {:.6} (그래디언트 폭발 가능성)", w2_grad_norm);
                        }
                    }

                    // 가중치 업데이트: w = w - η * grad_w
                    self.w1.sub_tensor(self.w1.grad().unwrap() * &lr).map_err(|e| {
                        error!("W1 가중치 업데이트 실패: {:?}", e);
                        e
                    })?;

                    self.w2.sub_tensor(self.w2.grad().unwrap() * &lr).map_err(|e| {
                        error!("W2 가중치 업데이트 실패: {:?}", e);
                        e
                    })?;

                    trace!("샘플 {} 가중치 업데이트 완료", m);

                    // 기울기 초기화
                    self.zero_grad().map_err(|e| {
                        error!("기울기 초기화 실패: {:?}", e);
                        e
                    })?;

                    trace!("샘플 {} 처리 완료", m);
                }
            }

            let epoch_duration = epoch_start.elapsed();

            // 샘플별 손실 통계 (디버그 레벨에서만)
            if !sample_errors.is_empty() {
                let avg_sample_loss = sample_errors.iter().sum::<f32>() / sample_errors.len() as f32;
                let max_sample_loss = sample_errors.iter().fold(0.0f32, |a, &b| a.max(b));
                let min_sample_loss = sample_errors.iter().fold(f32::INFINITY, |a, &b| a.min(b));

                debug!("에포크 {} 샘플 손실 통계: 평균={:.6}, 최대={:.6}, 최소={:.6}",
                   iter, avg_sample_loss, max_sample_loss, min_sample_loss);
            }

            // 1 epoch이 끝난 후 오차 재계산
            debug!("에포크 {} 전체 오차 재계산 중...", iter);
            let e_curr = self.compute_error(X, T).map_err(|e| {
                error!("에포크 {} 오차 계산 실패: {:?}", iter, e);
                e
            })?;

            resid = (e_curr - e_prev).abs();
            let improvement = e_prev - e_curr;

            // 성능 개선 여부 체크
            if e_curr < best_error {
                best_error = e_curr;
                best_epoch = iter;
                no_improvement_count = 0;
                debug!("새로운 최선 성능: {:.6} (에포크 {})", best_error, best_epoch);
            } else {
                no_improvement_count += 1;
                if no_improvement_count >= PATIENCE {
                    warn!("{}번 연속 개선 없음. 조기 종료 고려 중...", no_improvement_count);
                }
            }

            // 로깅 레벨에 따른 출력
            if iter % 10 == 0 || iter <= 5 || improvement < 0.0 {
                info!("에포크 {}: 오차={:.6}, 개선={:+.6}, 잔차={:.6}, 소요시간={:.2?}",
                 iter, e_curr, improvement, resid, epoch_duration);
            } else {
                debug!("에포크 {}: 오차={:.6}, 개선={:+.6}, 잔차={:.6}, 소요시간={:.2?}",
                  iter, e_curr, improvement, resid, epoch_duration);
            }

            // 학습 상태 체크
            if improvement < 0.0 {
                warn!("⚠️  오차가 증가했습니다! (이전: {:.6} -> 현재: {:.6})", e_prev, e_curr);
            } else if improvement < tol / 100.0 {
                debug!("개선이 매우 미미합니다: {:.6}", improvement);
            }

            // NaN 체크
            if e_curr.is_nan() || e_curr.is_infinite() {
                error!("오차가 NaN 또는 무한대입니다: {}. 훈련 중단.", e_curr);
                return Err(MlError::StringError("훈련 중 수치적 불안정성 발생".to_string()));
            }

            e_prev = e_curr;
            iter += 1;
        }

        let total_duration = start_time.elapsed();

        // 훈련 완료 상태 분석
        if resid < tol {
            info!("✅ 수렴 달성! 잔차 {:.6} < 허용오차 {:.6}", resid, tol);
            info!("최종 오차: {:.6} ({}번째 에포크에서 달성)", e_prev, iter - 1);
        } else if iter > max_iter {
            warn!("⚠️  최대 반복 횟수 도달. 잔차: {:.6}", resid);
            warn!("최종 오차: {:.6}, 최선 오차: {:.6} ({}번째 에포크)", e_prev, best_error, best_epoch);
        }

        info!("=== 훈련 완료 ===");
        info!("총 소요시간: {:.2?}", total_duration);
        info!("평균 에포크 시간: {:.2?}", total_duration / (iter - 1) as u32);
        info!("최종 성능: 오차={:.6}, 에포크={}", e_prev, iter - 1);

        if best_epoch != iter - 1 {
            info!("최선 성능: 오차={:.6}, 에포크={}", best_error, best_epoch);
        }

        Ok(())
    }

    /// 전체 데이터(X, T)에 대해 "평균 제곱 오차"를 계산
    fn compute_error(&self, X: &Vec<Arc<Variable>>, T: &Vec<Arc<Variable>>) -> MlResult<f32> {
        let mut sum_e = 0.0_f32;
        let n = X.len() as f32;

        for m in 0..X.len() {
            let (_z, y) = self.forward(&X[m])?;

            // diff = y - T[m]
            let y_data = y.tensor().data();
            let t_data = T[m].tensor().data();

            for i in 0..y_data.len() {
                let diff = y_data[i] - t_data[i];
                sum_e += diff * diff;
            }
        }

        Ok(sum_e / n)
    }

    #[cfg(feature = "enableBackpropagation")]
    fn zero_grad(&mut self) -> MlResult<()> {
        self.w1.clear_grad();
        self.w2.clear_grad();
        Ok(())
    }
}

// 사용 예시 테스트
mod tests {
    use std::io::Write;
    use super::*;
    use crate::tensor::TensorBase;
    use crate::var_input;
    use crate::var_with_label;
    use log::{info, debug, warn, error, trace};
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::Subscriber;
    use mnist::{MnistBuilder, Mnist};

    #[test]
    pub(crate) fn mlp_autograd_test() -> MlResult<()> {
        // 로거 초기화
        let _ = tracing_log::LogTracer::init();

        // 논블로킹(Non-blocking) writer를 설정합니다. 로그 I/O가 별도 스레드에서 처리됩니다.
        let (non_blocking_writer, _guard) = tracing_appender::non_blocking(std::io::stderr());

        // RUST_LOG 환경 변수를 사용하여 로그 레벨을 필터링합니다. (기존 env_logger와 동일한 기능)
        // 예: RUST_LOG=info cargo test
        // 설정되지 않은 경우 기본값으로 "trace" 레벨을 사용합니다.
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("Info"));

        // tracing subscriber를 구성하고 전역으로 설정합니다.
        let subscriber = Subscriber::builder()
            .with_writer(non_blocking_writer) // 논블로킹 writer 사용
            .with_env_filter(filter)          // 환경 변수 필터 사용
            .with_test_writer()               // `cargo test`에서 출력이 잘 보이도록 설정
            .finish();

        // tracing subscriber를 전역 로거로 설정합니다.
        let _ = tracing::subscriber::set_global_default(subscriber);


        info!("=== MLP 자동미분 테스트 시작 ===");

        let n_input = 784; // MNIST 이미지 크기
        let n_hidden = 30; // 은닉층 뉴런 개수
        let n_output = 10; // 출력층 뉴런 개수 (0-9 숫자 분류)

        info!("네트워크 구조: {}(입력) -> {}(은닉) -> {}(출력)", n_input, n_hidden, n_output);

        // MLP 생성
        let mut mlp = MLP::new(n_input, n_hidden, n_output);
        info!("MLP 모델이 성공적으로 생성되었습니다");

        let mut X = Vec::new();
        let mut T = Vec::new();

        info!("더미 데이터셋 생성 중...");

        // 각 클래스별로 몇 개씩 더미 데이터 생성
        for class in 0..10 {
            debug!("클래스 {} 데이터 생성 중", class);

            for sample_idx in 0..10 {  // 클래스당 10개 샘플
                // 784차원 랜덤 입력 (0-1 정규화)
                let mut input_data = vec![vec![0.0]; 784];
                for i in 0..784 {
                    input_data[i][0] = rand::random::<f32>();
                }
                let x = var_input!(Tensor::new(input_data));
                X.push(x);

                // 원-핫 인코딩된 타겟
                let mut target_data = vec![vec![0.0]; 10];
                target_data[class][0] = 1.0;
                let t = var_with_label!(Tensor::new(target_data), "target");
                T.push(t);

                trace!("클래스 {} 샘플 {} 생성 완료", class, sample_idx + 1);
            }
        }

        info!("데이터셋 생성 완료: 총 {}개 샘플 (클래스당 10개)", X.len());

        // 학습 파라미터 로깅
        let learning_rate = 0.05;
        let epochs = 1;
        let tolerance = 1e-6;

        info!("학습 파라미터:");
        info!("  - 학습률: {}", learning_rate);
        info!("  - 에포크: {}", epochs);
        info!("  - 허용 오차: {}", tolerance);

        info!("학습 시작...");
        let start_time = std::time::Instant::now();

        // 학습 (자동미분 사용)
        match mlp.train(&X, &T, learning_rate, epochs, tolerance) {
            Ok(_) => {
                let duration = start_time.elapsed();
                info!("학습 완료! 소요시간: {:.2?}", duration);
            },
            Err(e) => {
                error!("학습 중 오류 발생: {:?}", e);
                return Err(e);
            }
        }

        #[cfg(feature = "enableVisualization")]
        {
            info!("계산 그래프 시각화 생성 중...");
            match crate::tensor::VisualizationGraph::render_to_svg("graph/twolayer.svg") {
                Ok(_) => info!("SVG 그래프가 graph/twolayer.svg에 저장되었습니다"),
                Err(e) => warn!("SVG 그래프 저장 실패: {:?}", e),
            }

            match crate::tensor::VisualizationGraph::save_graph("graph/twolayer.dot") {
                Ok(_) => info!("DOT 그래프가 graph/twolayer.dot에 저장되었습니다"),
                Err(e) => warn!("DOT 그래프 저장 실패: {:?}", e),
            }
        }

        info!("모델 예측 테스트 중...");

        // 예측
        let test_input = &X[0];  // 첫 번째 샘플로 테스트
        debug!("첫 번째 샘플 (클래스 0)로 예측 테스트");

        match mlp.forward(test_input) {
            Ok((_z, y)) => {
                let prediction = y.tensor().data()[0];
                info!("예측 결과: {:.6}", prediction);

                // 전체 출력 확률 분포 로깅 (debug 레벨)
                let output_probs: Vec<f32> = y.tensor().data().to_vec();
                debug!("전체 출력 확률 분포: {:?}", output_probs);

                // 가장 높은 확률의 클래스 찾기
                let predicted_class = output_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap_or(0);

                info!("예측된 클래스: {} (확률: {:.4})", predicted_class, output_probs[predicted_class]);
                info!("실제 클래스: 0"); // 첫 번째 샘플은 클래스 0

                if predicted_class == 0 {
                    info!("✅ 예측 성공!");
                } else {
                    warn!("❌ 예측 실패 (예측: {}, 실제: 0)", predicted_class);
                }
            },
            Err(e) => {
                error!("예측 중 오류 발생: {:?}", e);
                return Err(e);
            }
        }

        info!("=== MLP 자동미분 테스트 완료 ===");
        Ok(())
    }
}