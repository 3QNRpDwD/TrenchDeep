
use crate::tensor::{OPERATOR_STORAGE, Tensor, TensorBase, AutogradFunction, ComputationGraph, GlobalFunction};
use crate::nn::activation::Sigmoid;
use crate::tensor::Variable;
use std::fmt;
use std::sync::Arc;
use crate::tensor::operators::{Add, Function, Matmul, Mul, Square, Sub, Sum};
use crate::{MlError, MlResult, scalar, var_with_label, var_input};
use crate::loss::{CrossEntropyLoss, MeanSquaredError};
use log::{info, debug, warn, error, trace};

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct ModelParameters {
    w1_data: Vec<f32>,
    w1_shape: Vec<usize>,
    b1_data: Vec<f32>,
    b1_shape: Vec<usize>,
    w2_data: Vec<f32>,
    w2_shape: Vec<usize>,
    b2_data: Vec<f32>,
    b2_shape: Vec<usize>,
}

pub struct MLP {
    pub w1: Arc<Variable>, // shape = [hidden_node, input_node]
    pub w2: Arc<Variable>, // shape = [output_node, hidden_node]
    pub b1: Arc<Variable>, // shape = [hidden_node, 1]
    pub b2: Arc<Variable>, // shape = [output_node, 1]
    // 활성화 함수를 MLP 구조체의 일부로 만들어 유연성 확보
    hidden_activation: GlobalFunction,
    output_activation: GlobalFunction,
}

impl fmt::Debug for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  w1.shape = {:?}, w2.shape = {:?}",
                 self.w1.tensor().shape(),

                 self.w2.tensor().shape())?;
        // 활성화 함수 정보 추가
        writeln!(f, "  hidden_activation = {}", self.hidden_activation.type_name())?;
        writeln!(f, "  output_activation = {}", self.output_activation.type_name())?;
        writeln!(f, "}}")
    }
}


impl MLP {
    /// n_input : 입력 뉴런 개수
    /// n_hidden: 은닉 뉴런 개수
    /// n_output: 출력 뉴런 개수
    /// hidden_activation: 은닉층에 적용할 활성화 함수
    /// output_activation: 출력층에 적용할 활성화 함수
    pub fn new(
        n_input: usize,
        n_hidden: usize,
        n_output: usize,
        hidden_activation: GlobalFunction,
        output_activation: GlobalFunction,
    ) -> Self {
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

        Self { w1, w2, b1, b2, hidden_activation, output_activation }
    }

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
        X: &Vec<Arc<Variable>>,
        T: &Vec<Arc<Variable>>,
        loss_function: GlobalFunction, // 손실 함수를 인자로 받음
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
        let mut e_prev = self.compute_total_error(X, T, &loss_function).map_err(|e| {
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
                ComputationGraph::reset_graph();
                let y = self.apply(x_m).map_err(|e| {
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
            let e_curr = self.compute_total_error(X, T, &loss_function).map_err(|e| {
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

    pub fn compute_total_error(&self, X: &Vec<Arc<Variable>>, T: &Vec<Arc<Variable>>, loss_function: &GlobalFunction) -> MlResult<f32> {
        let mut total_loss = 0.0;
        for m in 0..X.len() {
            let y = var_input!(self.forward(&X[m].tensor())?);
            let loss = loss_function.forward(&[&y.tensor(), &T[m].tensor()])?.remove(0);
            total_loss += loss.data()[0];
        }
        Ok(total_loss / X.len() as f32)
    }

    /// 모든 파라미터의 기울기를 0으로 초기화합니다.
    pub fn zero_grad(&mut self) -> MlResult<()> {
        self.w1.clear_grad();
        self.w2.clear_grad();
        self.b1.clear_grad();
        self.b2.clear_grad();
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
    use crate::loss::CrossEntropyLoss;
    use crate::nn::activation::Softmax;

    fn setup_logger() {
        use tracing_subscriber::{
            fmt::{self, format::FmtSpan},
            prelude::*,
            EnvFilter,
        };

        // 1. 파일에 로그를 저장하기 위한 '파일 Appender'를 설정합니다.
        // 'logs'라는 폴더 안에 'test_run.log'라는 이름으로 파일을 생성합니다.
        let file_appender = tracing_appender::rolling::minutely("logs", "test_run.log");
        let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);

        // 로그 레벨 필터 설정 (환경 변수가 없으면 'info' 레벨 사용)
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("Debug"));

        // 2. 파일로 출력하는 레이어(Layer)를 설정합니다.
        let file_layer = fmt::layer()
            .with_writer(non_blocking_appender)
            .with_ansi(true); // 파일에는 ANSI 색상 코드를 저장하지 않음

        // 3. 콘솔(stdout)로 출력하는 레이어를 설정합니다.
        let stdout_layer = fmt::layer()
            .with_writer(std::io::stdout);

        // 4. 설정된 레이어들을 조합하여 최종 로거를 초기화합니다.
        if tracing_subscriber::registry()
            .with(filter)
            .with(file_layer)
            .with(stdout_layer)
            .try_init()
            .is_err()
        {
            // 이미 로거가 설정된 경우(예: 다른 테스트에서 먼저 실행) 무시
        };

        // _guard를 반환하여 프로그램이 끝날 때까지 로그 파일이 열려 있도록 합니다.
        // 하지만 테스트 함수에서는 복잡해지므로, 여기서는 의도적으로 drop 시킵니다.
        // 대부분의 로그는 프로그램 종료 전에 기록되므로 큰 문제는 없습니다.
    }

    fn convert_to_variable_dataset(
        images: Vec<u8>,
        labels: Vec<u8>,
        num_items: usize,
        num_features: usize,
        num_classes: usize,
    ) -> MlResult<(Vec<Arc<Variable>>, Vec<Arc<Variable>>)> {
        let mut x_set = Vec::with_capacity(num_items);
        let mut t_set = Vec::with_capacity(num_items);

        // 이미지 데이터 처리 (u8 -> f32 정규화 및 Tensor 변환)
        let normalized_images: Vec<f32> = images.into_iter().map(|pixel| pixel as f32 / 255.0).collect();
        // 레이블 데이터 처리 (u8 -> f32 변환)
        let f32_labels: Vec<f32> = labels.into_iter().map(|label| label as f32).collect();

        for i in 0..num_items {
            let start_idx = i * num_features;
            let end_idx = start_idx + num_features;
            let image_slice = &normalized_images[start_idx..end_idx];
            let x = var_input!(Tensor::from_vec(image_slice.to_vec(), &[num_features, 1])?);
            x_set.push(x);

            let label_start_idx = i * num_classes;
            let label_end_idx = label_start_idx + num_classes;
            let label_slice = &f32_labels[label_start_idx..label_end_idx];
            let t = var_with_label!(Tensor::from_vec(label_slice.to_vec(), &[num_classes, 1])?, "target");
            t_set.push(t);
        }

        Ok((x_set, t_set))
    }

    #[test]
    fn mlp_mnist_classification_test() -> MlResult<()> {
        setup_logger();
        info!("=== MLP MNIST 분류 테스트 시작 ===");

        // --- 1. MNIST 데이터셋 로딩 ---
        let (n_train, n_val, n_features, n_classes) = (5000, 100, 784, 10);
        info!("MNIST 데이터셋 로딩 중... (학습: {}, 검증: {})", n_train, n_val);
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
         } = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(n_train as u32)
            .validation_set_length(0) // 별도 검증셋 사용 안 함
            .test_set_length(n_val as u32)
            .finalize();

        // --- 2. 데이터 전처리 및 모델 입력 형태로 변환 ---
        info!("데이터를 모델 입력 형식으로 변환 중...");
        let (x_train, t_train) = convert_to_variable_dataset(trn_img, trn_lbl, n_train, n_features, n_classes)?;
        let (x_test, t_test) = convert_to_variable_dataset(tst_img, tst_lbl, n_val, n_features, n_classes)?;
        info!("데이터 변환 완료.");

        // 네트워크 구조 정의
        let n_input = 784;
        let n_hidden = 30;
        let n_output = 10;

        // --- 리팩토링된 부분 ---
        // 1. 활성화 함수와 손실 함수를 명시적으로 생성
        // 은닉층: Sigmoid, 출력층: Softmax, 손실함수: CrossEntropy
        let hidden_activation = Sigmoid::new()?;
        let output_activation = Softmax::new()?;
        let loss_function = CrossEntropyLoss::new()?;

        info!("네트워크 구조: {}(입력) -> {}(은닉) -> {}(출력)", n_input, n_hidden, n_output);
        info!("활성화 함수: {} (은닉), {} (출력)", hidden_activation.name(), output_activation.name());

        // 2. MLP 생성 시 주입
        let mut mlp = MLP::new(n_input, n_hidden, n_output, hidden_activation, output_activation);
        info!("MLP 모델 생성 완료: {:?}", mlp);

        // 3. 학습 파라미터 조정
        let learning_rate = 0.05; // 더 안정적인 학습을 위해 학습률 감소
        let epochs = 10;
        let tolerance = 1e-4;    // 더 엄격한 수렴 조건

        info!("학습 파라미터: 학습률={}, 최대 에포크={}, 허용 오차={}", learning_rate, epochs, tolerance);

        // 4. train 함수 호출 시 손실 함수 전달
        mlp.train(&x_train, &t_train, loss_function, learning_rate, epochs, tolerance)?;

        // --- 4. 학습된 모델로 예측 및 정확도 평가 ---
        info!("=== 학습된 모델 평가 시작 (테스트셋 {}개) ===", n_val);
        let mut correct_predictions = 0;
        for i in 0..n_val {
            let test_input = &x_test[i];
            let true_label_tensor = &t_test[i];
            let y = mlp.apply(test_input)?;
            let output_probs = y.tensor().data();
            let predicted_class = output_probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let true_class = true_label_tensor.tensor().data().iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

            if predicted_class == true_class {
                correct_predictions += 1;
            }
        }

        let accuracy = correct_predictions as f32 / n_val as f32 * 100.0;
        info!("✅ 평가 완료: 정확도 = {:.2}%", accuracy);

        if accuracy > 80.0 {
            info!("🎉 목표 정확도 달성! 모델이 성공적으로 학습되었습니다.");

            let params_to_save = ModelParameters {
                w1_data: mlp.w1.tensor().data().to_vec(),
                w1_shape: mlp.w1.tensor().shape().to_vec(),
                b1_data: mlp.b1.tensor().data().to_vec(),
                b1_shape: mlp.b1.tensor().shape().to_vec(),
                w2_data: mlp.w2.tensor().data().to_vec(),
                w2_shape: mlp.w2.tensor().shape().to_vec(),
                b2_data: mlp.b2.tensor().data().to_vec(),
                b2_shape: mlp.b2.tensor().shape().to_vec(),
            };

            // --- 옵션 1: JSON 형태로 저장하기 (인간이 읽기 편함) ---
            match std::fs::File::create("model_parameters.json") {
                Ok(file) => {
                    if let Err(e) = serde_json::to_writer_pretty(file, &params_to_save) {
                        warn!("JSON 파일 저장 실패: {}", e);
                    } else {
                        info!("모델 파라미터를 'model_parameters.json' 파일로 저장했습니다.");
                    }
                }
                Err(e) => warn!("파일을 생성할 수 없습니다: {}", e),
            }
        } else {
            warn!("⚠️ 목표 정확도 미달. 하이퍼파라미터 튜닝이나 더 많은 학습이 필요할 수 있습니다.");
        }

        #[cfg(feature = "enableVisualization")]
        {
            info!("계산 그래프 시각화 생성 중...");
            match crate::tensor::VisualizationGraph::render_to_svg("graph/twolayer.svg") {
                Ok(_) => info!("SVG 그래프가 graph/twolayer.svg에 저장되었습니다"),
                Err(e) => warn!("SVG 그래프 저장 실패: {:?}", e),
            }
        }

        info!("=== MLP MNIST 테스트 완료 ===");
        Ok(())
    }
}