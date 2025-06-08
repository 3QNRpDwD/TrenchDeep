
use crate::tensor::{Tensor, TensorBase};
use crate::nn::activation::Sigmoid;
use crate::tensor::{Variable, AutogradFunction};
use std::fmt;
use std::sync::Arc;
use crate::tensor::operators::{Add, Function, Matmul, Square, Sub, Sum};
use crate::{MlResult, var_with_label};

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
                 unsafe { self.w1.tensor().shape() },
                 unsafe { self.w2.tensor().shape() })?;
        writeln!(f, "}}")
    }
}

impl MLP {
    /// n_input : 입력 뉴런 개수
    /// n_hidden: 은닉 뉴런 개수
    /// n_output: 출력 뉴런 개수
    pub fn new(n_input: usize, n_hidden: usize, n_output: usize) -> Self {
        // // w1: (hidden × (input+1)) 크기, rand 범위 [0,1) → [-0.1, +0.1) 로 변환
        // let w1_rand = Tensor::rand(&[n_hidden, n_input + 1]);
        // let w1_data: Vec<f32> = w1_rand.data().iter().map(|x| x * 0.2 - 0.1).collect();
        // let w1_tensor = Tensor::from_vec(w1_data, &[n_hidden, n_input + 1]).unwrap();
        // let w1 = Arc::new(Variable::new(w1_tensor));
        //
        // // w2: (output × (hidden+1)) 크기, 동일하게 초기화
        // let w2_rand = Tensor::rand(&[n_output, n_hidden + 1]);
        // let w2_data: Vec<f32> = w2_rand.data().iter().map(|x| x * 0.2 - 0.1).collect();
        // let w2_tensor = Tensor::from_vec(w2_data, &[n_output, n_hidden + 1]).unwrap();
        // let w2 = Arc::new(Variable::new(w2_tensor));
        //
        // MLP { w1, w2 }
        let w1_data: Vec<f32> = (0..n_hidden * n_input)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let w1 = var_with_label!(
            Tensor::from_vec(w1_data, &[n_hidden, n_input]).unwrap(),
            "w1"
        );

        let w2_data: Vec<f32> = (0..n_output * n_hidden)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let w2 = var_with_label!(
            Tensor::from_vec(w2_data, &[n_output, n_hidden]).unwrap(),
            "w2"
        );

        // bias 항들 초기화
        let b1_data: Vec<f32> = (0..n_hidden)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let b1 = var_with_label!(
            Tensor::from_vec(b1_data, &[n_hidden, 1]).unwrap(),
            "b1"
        );

        let b2_data: Vec<f32> = (0..n_output)
            .map(|_| rand::random::<f32>() * 0.5 - 0.25)
            .collect();
        let b2 = var_with_label!(
            Tensor::from_vec(b2_data, &[n_output, 1]).unwrap(),
            "b2"
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
        let y = sigmoid.apply(&[&uo])?;

        Ok((ah, y)) // 은닉층 출력과 최종 출력 반환
    }

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

        // 초기 오차 계산
        let mut e_prev = self.compute_error(X, T)?;
        println!("{}-th update and error is {}", iter - 1, e_prev);

        while resid >= tol && iter <= max_iter {
            // 1 epoch 동안 샘플별로 순전파→역전파→업데이트
            for m in 0..n_samples {
                crate::tensor::ComputationGraph::reset_graph();
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
                    self.update_weights(eta)?;

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
            let y_data = unsafe { y.tensor().data() };
            let t_data = unsafe { T[m].tensor().data() };

            for i in 0..y_data.len() {
                let diff = y_data[i] - t_data[i];
                sum_e += diff * diff;
            }
        }

        Ok(sum_e / n)
    }

    fn sum_variable(&self, a: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        // Sum all elements to scalar
        let a_data = unsafe { a.tensor().data() };
        let sum: f32 = a_data.iter().sum();

        let result_tensor = Tensor::from_vec(vec![sum], &[1, 1]).unwrap();
        Ok(Arc::new(Variable::new(result_tensor)))
    }

    #[cfg(feature = "enableBackpropagation")]
    fn update_weights(&mut self, eta: f32) -> MlResult<()> {
        // w1 업데이트
        if let Some(grad_w1) = self.w1.grad() {
            let w1_data = unsafe { self.w1.tensor().data() };
            let grad_data = grad_w1.data();
            let shape = unsafe { self.w1.tensor().shape() };

            let updated_data: Vec<f32> = w1_data.iter().zip(grad_data.iter())
                .map(|(w, g)| w - eta * g)
                .collect();

            let updated_tensor = Tensor::from_vec(updated_data, shape).unwrap();
            self.w1 = Arc::new(Variable::new(updated_tensor));
        }

        // w2 업데이트
        if let Some(grad_w2) = self.w2.grad() {
            let w2_data = unsafe { self.w2.tensor().data() };
            let shape = unsafe { self.w2.tensor().shape() };
            let grad_data = grad_w2.data();
            

            let updated_data: Vec<f32> = w2_data.iter().zip(grad_data.iter())
                .map(|(w, g)| w - eta * g)
                .collect();

            let updated_tensor = Tensor::from_vec(updated_data, shape).unwrap();
            self.w2 = Arc::new(Variable::new(updated_tensor));
        }

        Ok(())
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

    #[test]
    pub(crate) fn mlp_autograd_test() -> MlResult<()> {
        use crate::var_with_label;
        let n_input = 2;
        let n_hidden = 3;
        let n_output = 1;

        // MLP 생성
        let mut mlp = MLP::new(n_input, n_hidden, n_output);

        // 입력 데이터를 Variable로 래핑
        let x1 = var_with_label!(Tensor::new(vec![vec![0.0], vec![0.0]]), "input_1");
        let x2 = var_with_label!(Tensor::new(vec![vec![1.0], vec![0.0]]), "input_2");
        let x3 = var_with_label!(Tensor::new(vec![vec![0.0], vec![1.0]]), "input_3");
        let x4 = var_with_label!(Tensor::new(vec![vec![1.0], vec![1.0]]), "input_4");
        let X = vec![x1, x2, x3, x4];

        // 타겟 데이터를 Variable로 래핑
        let t1 = var_with_label!(Tensor::new(vec![vec![0.0]]), "target_1");
        let t2 = var_with_label!(Tensor::new(vec![vec![1.0]]), "target_2");
        let t3 = var_with_label!(Tensor::new(vec![vec![0.0]]), "target_3");
        let t4 = var_with_label!(Tensor::new(vec![vec![1.0]]), "target_4");
        let T = vec![t1, t2, t3, t4];

        // 학습 (자동미분 사용)
        mlp.train(&X, &T, 0.05, 100, 1e-6)?;

        // 예측
        let test_input = &X[0];
        let (_z, y) = mlp.forward(test_input)?;
        crate::tensor::VisualizationGraph::render_to_svg("graph/twolayer.svg").unwrap();
        crate::tensor::VisualizationGraph::save_graph("graph/twolayer.dot").unwrap();

        let prediction = unsafe { y.tensor().data()[0] };
        println!("Prediction for [0,0]: {}", prediction);

        Ok(())
    }
}