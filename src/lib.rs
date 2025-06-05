pub mod tensor;
pub mod backend;
pub mod nn;
pub mod optimizer;
pub mod loss;

#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    InvalidDataLength {
        expected: usize,
        got: usize,
    },
    InvalidOperation {
        op: &'static str,
        reason: String,
    },
    InvalidAxis {
        axis: usize,
        shape: Vec<usize>,
    },
    MatrixMultiplicationError {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    EmptyTensor,
}

impl std::error::Error for TensorError {}

#[derive(Debug)]
pub enum MlError {
    TensorError(TensorError),
    StringError(String),
}

impl std::error::Error for MlError {}

impl From<TensorError> for MlError {
    fn from(error: TensorError) -> Self {
        MlError::TensorError(error)
    }
}
impl From<MlError> for TensorError {
    fn from(val: MlError) -> Self {
        match val {
            MlError::TensorError(e) => e,
            _ => unreachable!(),
        }
    }
}

impl From<String> for MlError {
    fn from(error: String) -> Self {
        MlError::StringError(error)
    }
}

impl From<&str> for MlError {
    fn from(error: &str) -> Self {
        MlError::StringError(error.to_string())
    }
}

pub type MlResult<T> = Result<T, MlError>;

pub mod mlp {
    use crate::tensor::{Tensor, TensorBase};
    use crate::nn::activation::Sigmoid;
    use std::fmt;
    use crate::scalar_ops;
    use crate::tensor::operators::{Function, Matmul, Sum, Transpose};
    pub struct MLP {
        pub w1: Tensor<f32>, // shape = [hidden_node, input_node + 1]
        pub w2: Tensor<f32>, // shape = [output_node, hidden_node + 1]
    }

    impl fmt::Debug for MLP {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "MLP {{")?;
            writeln!(f, "  w1.shape = {:?}, w2.shape = {:?} ", self.w1.shape(), self.w2.shape())?;
            writeln!(f, "}}")
        }
    }

    impl MLP {
        /// n_input : 입력 뉴런 개수
        /// n_hidden: 은닉 뉴런 개수
        /// n_output: 출력 뉴런 개수
        pub fn new(n_input: usize, n_hidden: usize, n_output: usize) -> Self {
            // w1: (hidden × (input+1)) 크기, rand 범위 [0,1) → [-0.1, +0.1) 로 변환
            let w1_rand = Tensor::rand(&[n_hidden, n_input + 1]);
            let w1_data: Vec<f32> = w1_rand.data().iter().map(|x| x * 0.2 - 0.1).collect();
            let w1 = Tensor::from_vec(w1_data, &[n_hidden, n_input + 1]).unwrap();

            // w2: (output × (hidden+1)) 크기, 동일하게 초기화
            let w2_rand = Tensor::rand(&[n_output, n_hidden + 1]);
            let w2_data: Vec<f32> = w2_rand.data().iter().map(|x| x * 0.2 - 0.1).collect();
            let w2 = Tensor::from_vec(w2_data, &[n_output, n_hidden + 1]).unwrap();

            MLP { w1, w2 }
        }

        /// 단일 샘플 x에 대해 순전파 수행
        ///
        /// x: Tensor of shape [input_node, 1]
        /// 반환: (z, y)
        ///   - z: 은닉층 활성화값( bias row 포함 ) → shape = [(hidden_node+1), 1]
        ///        z[(0,0)] = 1.0 (bias), z[(1..,0)] = sigmoid(u_h)
        ///   - y: 출력층 활성화값 → shape = [output_node, 1]
        pub fn forward(&self, x: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
            let sigmoid = Sigmoid::new().unwrap();
            let matmul = Matmul::new().unwrap();
            // 1) x 의 shape = [n_input, 1] 이라고 가정
            let n_input = x.shape()[0];
            let n_hidden = self.w1.shape()[0];

            // 2) 입력벡터에 bias 항 추가 → xl shape = [n_input+1, 1],
            let mut xl = Tensor::ones(&[n_input + 1, 1]);
            // 2) 입력벡터에 bias 항 추가 → xl shape = [n_input+1, 1],
            let mut xl_data = vec![1.0];
            xl_data.extend(x.data().iter());
            let xl = Tensor::from_vec(xl_data, &[n_input + 1, 1]).unwrap();

            // 3) 은닉층 입력 u_h = w1.matmul(xl) → shape = [hidden, 1]
            let uh: Tensor<f32> = matmul.forward(&[&self.w1, &xl]).unwrap().remove(0);

            // 4) 은닉층 활성화 a_h = sigmoid(u_h) → shape = [hidden, 1]
            let a_h = sigmoid.forward(&[&uh]).unwrap().remove(0);

            // bias 포함 z 벡터 생성 → shape = [hidden+1, 1], z[(0,0)] = 1, z[(1+i,0)] = a_h[(i,0)]
            let mut z = Tensor::ones(&[n_hidden + 1, 1]);
            let mut z_data = vec![1.0];
            z_data.extend(a_h.data().iter());
            let z = Tensor::from_vec(z_data, &[n_hidden + 1, 1]).unwrap();

            // 5) 출력층 입력 u_o = w2.matmul(z) → shape = [output, 1]
            let uo: Tensor<f32> = matmul.forward(&[&self.w2, &z]).unwrap().remove(0);

            // 6) 출력층 활성화 y = sigmoid(uo) → shape = [output, 1]
            let y = sigmoid.forward(&[&uo]).unwrap().remove(0);

            (z, y)
        }

        pub fn train(
            &mut self,
            X: &Vec<Tensor<f32>>,
            T: &Vec<Tensor<f32>>,
            eta: f32,
            max_iter: usize,
            tol: f32,
        ) {
            let sigmoid = Sigmoid::new().unwrap();
            let matmul = Matmul::new().unwrap();
            let transpose = Transpose::new().unwrap();
            let n_samples = X.len();
            let mut resid = tol * 2.0;
            let mut iter = 1;

            // 초기 오차 계산 (E_prev)
            let mut e_prev = self.compute_error(X, T);
            println!("{}-th update and error is {}", iter - 1, e_prev);

            while resid >= tol && iter <= max_iter {
                // 1 epoch 동안 “샘플 단위”로 순전파→역전파→업데이트
                for m in 0..n_samples {
                    let x_m = &X[m]; // shape [n_input, 1]
                    let t_m = &T[m]; // shape [n_output, 1]

                    // === 순전파 ===
                    // (z, y) 반환
                    let (z, y) = self.forward(x_m);

                    // === 출력층 δ_k 계산 ===
                    let n_output = y.shape()[0];
                    let d_sig_o = &y * &(Tensor::ones(&[n_output, 1]) - &y);
                    // δ_k = (y - t) ∘ d_sig_o  (∘ : element-wise 곱)
                    let delta_k = (&y - t_m) * &d_sig_o; // shape [output, 1]

                    // === 출력층 가중치 기울기 dw2 계산 ===
                    let zt = transpose.forward(&[&z]).unwrap().remove(0); // shape [1, hidden+1]
                    let dw2 = matmul.forward(&[&delta_k, &zt])
                        .unwrap()
                        .remove(0); // shape [output, hidden+1]

                    // === 은닉층 δ_j 계산 ===
                    let w2_shape = self.w2.shape();
                    let mut w2_no_bias_data = Vec::with_capacity(w2_shape[0] * (w2_shape[1] - 1));
                    for row in 0..w2_shape[0] {
                        for col in 1..w2_shape[1] {
                            w2_no_bias_data.push(self.w2.data()[row * w2_shape[1] + col]);
                        }
                    }
                    let w2_no_bias = Tensor::from_vec(w2_no_bias_data, &[w2_shape[0], w2_shape[1] - 1]).unwrap();
                    // (w2[:,1..]).T ⋅ delta_k  → shape [hidden, 1]
                    let tmp = matmul.forward(&[&transpose.forward(&[&w2_no_bias]).unwrap().remove(0), &delta_k])
                        .unwrap()
                        .remove(0); // shape [hidden, 1]

                    // sigmoid'(u_h) = a_h ∘ (1 - a_h) 을 다시 구하려면 u_h를 재계산
                    //   u_h = w1 ⋅ xl (xl: bias 포함 입력)
                    let n_input = x_m.shape()[0];
                    // xl: [n_input+1, 1]
                    // xl: [n_input+1, 1] (bias 포함 입력 텐서 새로 생성)
                    let mut xl_data = vec![1.0];
                    xl_data.extend(x_m.data().iter());
                    let xl = Tensor::from_vec(xl_data, &[n_input + 1, 1]).unwrap();

                    let uh = matmul.forward(&[&self.w1, &xl]).unwrap().remove(0); // shape [hidden, 1]
                    let n_hidden = uh.shape()[0];
                    let a_h: Tensor<f32> = sigmoid.forward(&[&uh]).unwrap().remove(0);       // shape [hidden, 1]

                    // d_sig_h: sigmoid'(u_h) = a_h * (1 - a_h)
                    // d_sig_h: sigmoid'(u_h) = a_h * (1 - a_h)
                    let d_sig_h = &a_h * &(Tensor::ones(&[n_hidden, 1]) - &a_h);
                    // δ_j = tmp ∘ d_sig_h  (element-wise 곱) → shape [hidden, 1]
                    let delta_j: Tensor<f32> = tmp * d_sig_h;

                    // === 은닉층 가중치 기울기 dw1 계산 ===
                    // dw1 = delta_j ⋅ xlᵀ   → [hidden,1] ⋅ [1, input+1] = [hidden, input+1]
                    let dw1 = matmul.forward(&[&delta_j, &transpose.forward(&[&xl]).unwrap().remove(0)])
                        .unwrap()
                        .remove(0); // shape [hidden, input+1]

                    // === 가중치, bias 업데이트 ===
                    // Python: v = v - η⋅dw2, w = w - η⋅dw1
                    self.w2 = &self.w2 - scalar_ops!(dw2, Mul, &eta).unwrap();
                    self.w1 = &self.w1 - scalar_ops!(dw1, Mul, &eta).unwrap();
                }

                // 1 epoch이 끝난 후 오차 재계산
                let e_curr = self.compute_error(X, T);
                resid = (e_curr - e_prev).abs();
                e_prev = e_curr;
                println!("{}-th update and error is {}", iter, e_curr);
                iter += 1;
            }

            println!("The learning is finished");
        }

        /// 전체 데이터(X, T)에 대해 “평균 제곱 오차”를 계산
        ///
        /// Python: E = Σ (y - t)²  → E / N
        fn compute_error(&self, X: &Vec<Tensor<f32>>, T: &Vec<Tensor<f32>>) -> f32 {
            let mut sum_e = 0.0_f32;
            let n = X.len() as f32;

            for m in 0..X.len() {
                let (_z, y) = self.forward(&X[m]);
                // diff = y – T[m], shape [output,1]
                let diff = &y - &T[m];
                // 제곱 후 모두 더하기
                let sq = &diff * &diff;
                let sq_data = sq.data();
                for v in sq_data {
                    sum_e += *v;
                }
            }

            sum_e / n
        }
    }
}

#[cfg(test)]
mod benchmark {
    use crate::tensor::operators::{Add, Function, Mul, Pow, Square, Sub};
    use crate::tensor::{Tensor, TensorBase, Variable, AutogradFunction};
    use crate::{scalar, var_input, variable, MlResult};
    use std::sync::Arc;

    fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        if tensor.shape() != expected_tensor.shape() {
            return Err("Shape mismatch".into());
        }

        let tensor_data = tensor.data();
        let expected_data = expected_tensor.data();

        for (t, e) in tensor_data.iter().zip(expected_data.iter()) {
            if (t - e).abs() > 1e-6 {
                return Err("Data mismatch".into());
            }
        }

        Ok(())
    }

    fn sphere_function(x: &Arc<Variable<f32>>, y: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        let mut pow = Pow::new()?;
        let add = Add::new()?;
        pow.power = Some(2.0);

        add.apply(&[
            &pow.apply(&[x])?,
            &pow.apply(&[y])?]
        )
    }

    fn matyas_function(x: &Arc<Variable<f32>>, y: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        let sub = Sub::new()?;
        let mul = Mul::new()?;
        let O_26 = Arc::new(variable!(vec![vec![0.26]]));
        let O_48 = Arc::new(variable!(vec![vec![0.48]]));

        let sphere = sphere_function(x, y)?;
        sub.apply(&[                   // (0.26 * sphere) - (0.48 * x * y)
            &mul.apply(&[&O_26, &sphere])?,                     // 0.26 * sphere
            &mul.apply(&[&O_48, &mul.apply(&[x, y])?])?  // 0.48 * x * y
        ])
    }

    fn goldstein_price_function(x: &Arc<Variable<f32>>, y: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        // Helper function to create constant variables
        fn constant(value: f32) -> Arc<Variable<f32>> {
            Arc::new(variable!(vec![vec![value]]))
        }

        let add = Add::new()?;
        let square = Square::new()?;
        let mul = Mul::new()?;
        let sub = Sub::new()?;

        // Define constants
        let num_1   = constant(1.0);
        let num_2   = constant(2.0);
        let num_3   = constant(3.0);
        let num_6   = constant(6.0);
        let num_12  = constant(12.0);
        let neg_14  = constant(-14.0);
        let neg_32  = constant(-32.0);
        let neg_36  = constant(-36.0);

        // Compute a = x + y + 1
        let a =
            add.apply(&[
                &add.apply(&[x, y])?,
                &num_1
            ])?;

        // Compute x squared and y squared
        let x_squared = square.apply(&[x])?;
        let y_squared = square.apply(&[y])?;
        // Compute b = (((((19 - 14x) + 3x^2) - 14y) + 6xy) + 3y^2)
        let term2_b = mul.apply(&[&neg_14, x])?;
        let term3_b = mul.apply(&[&num_3, &x_squared])?;
        let term4_b = mul.apply(&[&neg_14, y])?;
        let term5_b = mul.apply(&[&num_6, &mul.apply(&[x, y])?])?;
        let term6_b = mul.apply(&[&num_3, &y_squared])?;

        let b =
            add.apply(&[
                &add.apply(&[
                    &add.apply(&[
                        &add.apply(&[
                            &add.apply(&[&constant(19.0), &term2_b])?,
                            &term3_b])?,
                        &term4_b])?,
                    &term5_b])? ,
                &term6_b
            ])?; // (((((19 - 14x) + 3x^2) - 14y) + 6xy) + 3y^2)

        // Compute first part: 1 + (a^2 * b)
        let a_squared   = square.apply(&[&a])?;
        let a_squared_b = mul.apply(&[&a_squared, &b])?;
        let first_part  = add.apply(&[&num_1, &a_squared_b])?;

        // Compute c = 2x - 3y
        let two_x   = mul.apply(&[&num_2, x])?;
        let three_y = mul.apply(&[&num_3, y])?;
        let c       = sub.apply(&[&two_x, &three_y])?;

        // Compute d = 18 - 32x + 12x^2 + 48y - 36xy + 27y^2
        let term2_d = mul.apply(&[&neg_32, x])?;
        let term3_d = mul.apply(&[&num_12, &x_squared])?;
        let term4_d = mul.apply(&[&constant(48.0), y])?;
        let term5_d = mul.apply(&[&neg_36, &mul.apply(&[x, y])?])?;
        let term6_d = mul.apply(&[&constant(27.0), &y_squared])?;

        let d =
            add.apply(&[
                &add.apply(&[
                    &add.apply(&[
                        &add.apply(&[
                            &add.apply(&[&constant(18.0), &term2_d])?,
                            &term3_d])?,
                        &term4_d])?,
                    &term5_d])? ,
                &term6_d])?; // 18 - 32x + 12x^2 + 48y - 36xy + 27y^2

        // Compute second part: 30 + c^2 * d
        let c_squared   = square.apply(&[&c])?;
        let c_squared_d = mul.apply(&[&c_squared, &d])?;
        let second_part = add.apply(&[&constant(30.0), &c_squared_d])?;

        // Compute final function value
        mul.apply(&[&first_part, &second_part])
    }

    fn rosenbrock_function(x0: &Arc<Variable<f32>>, x1: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        let sub = Sub::new()?;
        let add = Add::new()?;
        let square = Square::new()?;
        let mul = Mul::new()?;

        add.apply(&[
            &mul.apply(&[
                &Arc::new(variable!(vec![vec![100.0]])),
                &square.apply(&[
                    &sub.apply(&[
                        &x1,
                        &square.apply(&[&x0])?])?
                ])?
            ])?,
            &square.apply(&[
                &sub.apply(&[
                    &Arc::new(variable!(vec![vec![1.0]])),
                    &x0
                ])?
            ])?
        ])
    }

    #[test]
    fn sphere() -> MlResult<()> {
        let x = Arc::new(variable!(vec![vec![1.0]]));
        let y = Arc::new(variable!(vec![vec![1.0]]));
        let z = sphere_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
            assert_tensor_eq(&y.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
        }
        Ok(())
    }

    #[test]
    fn matyas() -> MlResult<()> {
        let x = Arc::new(variable!(vec![vec![1.0]]));
        let y = Arc::new(variable!(vec![vec![1.0]]));
        let z = matyas_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        z.backward()?;
        Ok(())
    }

    #[test]
    fn goldstein() -> MlResult<()> {
        let x = Arc::new(variable!(vec![1.0], &[1,1], "x"));
        let y = Arc::new(variable!(vec![1.0], &[1,1], "y"));
        let z = goldstein_price_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![-5376.0]]))?;
            assert_tensor_eq(&y.grad().unwrap(), &Tensor::new(vec![vec![8064.0]]))?;
        }

        #[cfg(feature = "enableVisualization")]
        crate::tensor::VisualizationGraph::save_graph("graph/goldstein.dot").unwrap();
        Ok(())
    }

    #[test]
    fn rosenbrock() -> MlResult<()> {
        let x0 = Arc::new(variable!(vec![0.0], &[1,1], "x0"));
        let x1 = Arc::new(variable!(vec![2.0], &[1,1], "x1"));

        let y = rosenbrock_function(&x0, &x1)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_tensor_eq(&x0.grad().unwrap(), &Tensor::new(vec![vec![-2.0]]))?;
            assert_tensor_eq(&x1.grad().unwrap(), &Tensor::new(vec![vec![400.0]]))?;
        }

        #[cfg(feature = "enableVisualization")]
        crate::tensor::VisualizationGraph::save_graph("graph/rosenbrock.dot").unwrap();
        Ok(())
    }

    #[test]
    #[cfg(feature = "enableBackpropagation")]
    fn rosenbrock_gradient_descent_function() -> MlResult<()> {
        let x0 = var_input!(Tensor::new(vec![vec![0.0]]));
        let x1 = var_input!(Tensor::new(vec![vec![2.0]]));
        let y = rosenbrock_function(&x0, &x1)?;
        let iter: usize = 1000;
        let learning_rate = scalar!(0.001);

        for i in 0..iter { // 0부터
            let y = rosenbrock_function(&x0, &x1)?;
            y.backward()?;

            //파라미터 갱신
            x0.swap_tensor( unsafe { x0.tensor() } - &x0.grad().unwrap() * &learning_rate );
            x1.swap_tensor( unsafe { x1.tensor() } - &x1.grad().unwrap() * &learning_rate );

            #[cfg(feature = "debugging")]
            {
                if i % 1000 == 0 {
                    println!(
                        "iter - {}\n\
                [ x0.tensor: {:?}, x0.grad: {:?} ]\n\
                [ x1.tensor: {:?}, x1.grad: {:?} ]"
                        , i, unsafe { x0.tensor() }, x0.grad(), unsafe { x1.tensor() }, x1.grad()
                    );
                }
            }

            //파라미터 갱신
            x0.sub_tensor(&x0.grad().unwrap() * &learning_rate);
            x1.sub_tensor(&x1.grad().unwrap() * &learning_rate);
        }
        Ok(())
    }

    #[test]
    pub(crate) fn mlp_mnist_like_test() -> MlResult<()> {
        use crate::mlp::MLP;
        use crate::tensor::TensorBase;

        // Python과 동일한 설정
        let n_input = 784;  // 28x28
        let n_hidden = 30;
        let n_output = 10;

        // MLP 생성 (동일한 시드 사용하도록 수정 필요)
        let mut mlp = MLP::new(n_input, n_hidden, n_output);

        // 간단한 더미 MNIST 데이터 생성 (실제로는 MNIST 로드 필요)
        let mut X = Vec::new();
        let mut T = Vec::new();

        // 각 클래스별로 몇 개씩 더미 데이터 생성
        for class in 0..10 {
            for _ in 0..10 {  // 클래스당 10개 샘플
                // 784차원 랜덤 입력 (0-1 정규화)
                let mut input_data = vec![vec![0.0]; 784];
                for i in 0..784 {
                    input_data[i][0] = rand::random::<f32>();
                }
                let x = Tensor::new(input_data);
                X.push(x);

                // 원-핫 인코딩된 타겟
                let mut target_data = vec![vec![0.0]; 10];
                target_data[class][0] = 1.0;
                let t = Tensor::new(target_data);
                T.push(t);
            }
        }

        // Python과 동일한 하이퍼파라미터로 학습
        mlp.train(&X, &T, 0.05, 500, 1e-10);

        // 예측 테스트
        let test_input = &X[0];  // 첫 번째 샘플로 테스트
        let (_z, y) = mlp.forward(test_input);

        // argmax로 예측 클래스 찾기
        let data = y.data();
        let mut predicted_class = 0;
        let mut max_prob = data[0];
        for (idx, &prob) in data.iter().enumerate().skip(1) {
            if prob > max_prob {
                max_prob = prob;
                predicted_class = idx;
            }
        }

        println!("Predicted class: {}, Max probability: {}", predicted_class, max_prob);
        Ok(())
    }

    // 이 코드는 Python으로 구현된 2층 딥러닝 모델과 정확히 비교하기 위한 테스트입니다.
    #[test]
    pub(crate) fn mlp_exact_python_comparison() -> MlResult<()> {
        use crate::mlp::MLP;
        use crate::tensor::TensorBase;

        // 작은 규모로 Python 코드와 정확히 비교
        let n_input = 4;
        let n_hidden = 3;
        let n_output = 2;

        let mut mlp = MLP::new(n_input, n_hidden, n_output);

        // 고정된 값으로 테스트 (시드 고정 효과)
        let x1 = Tensor::new(vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]]);
        let x2 = Tensor::new(vec![vec![0.5], vec![0.6], vec![0.7], vec![0.8]]);
        let X = vec![x1, x2];

        let t1 = Tensor::new(vec![vec![1.0], vec![0.0]]);
        let t2 = Tensor::new(vec![vec![0.0], vec![1.0]]);
        let T = vec![t1, t2];

        // Python과 동일한 학습률
        mlp.train(&X, &T, 0.05, 100, 1e-10);

        // 순전파 과정 상세 출력으로 Python과 비교
        let (_z, y) = mlp.forward(&X[0]);
        println!("Forward pass result: {:?}", y.data());

        Ok(())
    }
}