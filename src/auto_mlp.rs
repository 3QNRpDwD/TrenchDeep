
use crate::tensor::{OPERATOR_STORAGE, Tensor, TensorBase, AutogradFunction};
use crate::nn::activation::Sigmoid;
use crate::tensor::Variable;
use std::fmt;
use std::sync::Arc;
use crate::tensor::operators::{Add, Function, Matmul, Mul, Square, Sub, Sum};
use crate::{MlResult, scalar, var_with_label};

pub struct MLP {
    pub w1: Arc<Variable<f32>>, // shape = [hidden_node, input_node + 1]
    pub w2: Arc<Variable<f32>>, // shape = [output_node, hidden_node + 1]
    pub b1: Arc<Variable<f32>>, // shape = [hidden_node, 1]
    pub b2: Arc<Variable<f32>>, // shape = [output_node, 1]
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
    pub fn forward(&self, x: &Arc<Variable<f32>>) -> MlResult<(Arc<Variable<f32>>, Arc<Variable<f32>>)> {
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
        X: &Vec<Arc<Variable<f32>>>,
        T: &Vec<Arc<Variable<f32>>>,
        eta: f32,
        max_iter: usize,
        tol: f32,
    ) -> MlResult<()> {
        let sub = Sub::new()?;
        let square = Square::new()?;
        let sum = Sum::new()?;
        let n_samples = X.len();
        let mut resid = tol * 2.0;
        let mut iter = 1;
        let lr = scalar!(eta);

        // 초기 오차 계산
        let mut e_prev = self.compute_error(X, T)?;
        println!("{}-th update and error is {}", iter - 1, e_prev);

        while resid >= tol && iter <= max_iter {
            // 1 epoch 동안 샘플별로 순전파→역전파→업데이트
            for m in 0..n_samples {
                let x_m = &X[m];
                let t_m = &T[m];

                // === 순전파 (자동미분 그래프 구성) ===
                let (_z, y) = self.forward(x_m)?;
                // y는 출력층의 활성화값, shape = [output_node, 1]
                // t_m은 타겟값, shape = [output_node, 1]

                #[cfg(feature = "enableBackpropagation")]
                {
                // === 손실 함수 계산 ===
                // loss = sum((y - t)²) / 2  -> MSE loss
                let diff = sub.apply(&[&y, t_m])?;
                let squared = square.apply(&[&diff])?;
                let loss = sum.apply(&[&squared])?;

                // === 역전파 (자동미분) ===
                    // loss에서 시작해서 모든 매개변수에 대한 기울기 계산
                    loss.backward()?;

                    // 가중치 업데이트: w = w - η * grad_w
                    Mul::new()?;
                    self.w1.sub_tensor(self.w1.grad().unwrap() * &lr);
                    self.w2.sub_tensor(self.w2.grad().unwrap() * &lr);

                    // 기울기 초기화
                    self.zero_grad()?;
                }
            }

            // 1 epoch이 끝난 후 오차 재계산
            let e_curr = self.compute_error(X, T)?;
            resid = (e_curr - e_prev).abs();
            e_prev = e_curr;
            println!("{}-th update and error is {}", iter, e_curr);
            iter += 1;
        }

        println!("The learning is finished");
        Ok(())
    }

    /// 전체 데이터(X, T)에 대해 "평균 제곱 오차"를 계산
    fn compute_error(&self, X: &Vec<Arc<Variable<f32>>>, T: &Vec<Arc<Variable<f32>>>) -> MlResult<f32> {
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorBase;
    use crate::var_input;
    use crate::var_with_label;

    #[test]
    #[cfg(feature = "enableBackpropagation")]
    pub(crate) fn mlp_autograd_test() -> MlResult<()> {
        let n_input = 784; // MNIST 이미지 크기
        let n_hidden = 30; // 은닉층 뉴런 개수
        let n_output = 10; // 출력층 뉴런 개수 (0-9 숫자 분류)

        // MLP 생성
        let mut mlp = MLP::new(n_input, n_hidden, n_output);
        
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
                let x = var_input!(Tensor::new(input_data));
                X.push(x);
        
                // 원-핫 인코딩된 타겟
                let mut target_data = vec![vec![0.0]; 10];
                target_data[class][0] = 1.0;
                let t = var_with_label!(Tensor::new(target_data), "target");
                T.push(t);
            }
        }

        // 학습 (자동미분 사용)
        mlp.train(&X, &T, 0.05, 1, 1e-6)?;

        #[cfg(feature = "enableVisualization")]
        {
            crate::tensor::VisualizationGraph::render_to_svg("graph/twolayer.svg").unwrap();
            crate::tensor::VisualizationGraph::save_graph("graph/twolayer.dot").unwrap();
        }

        // 예측
        let test_input   = &X[0];  // 첫 번째 샘플로 테스트
        let (_z, y) = mlp.forward(test_input)?;


        let prediction = y.tensor().data()[0];
        println!("Prediction for &X[0]: {}", prediction);

        Ok(())
    }
}