use super::*;

impl Function for Softmax {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Softmax)
    }

    /// Softmax 함수의 순전파를 계산합니다.
    /// S(x_i) = exp(x_i) / sum(exp(x_j))
    fn forward(&mut self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input = targets[0];
        let input_data = input.data();

        // 1. 수치 안정성(Numerical Stability)을 위한 처리
        // 입력값에서 최댓값을 빼주어 exp() 계산 시 오버플로우를 방지합니다.
        // 이 과정은 최종 결과에 영향을 주지 않습니다.
        let max_val = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // 2. 각 요소에 exp()를 적용합니다.
        let exp_values: Vec<f32> = input_data.iter().map(|&x| (x - max_val).exp()).collect();

        // 3. exp()가 적용된 모든 요소의 합을 구합니다.
        let sum_of_exps: f32 = exp_values.iter().sum();

        // 4. 각 exp() 값을 합계로 나누어 최종 확률을 계산합니다.
        let softmax_output: Vec<f32> = exp_values.iter().map(|&exp_val| exp_val / sum_of_exps).collect();

        Ok(vec![GlobalTensor::from_vec(softmax_output, input.shape())?])
    }

    /// Softmax 함수의 역전파(gradient)를 계산합니다.
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&mut self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        // targets[0]는 순전파 시의 Softmax 출력(y)입니다.
        // grad는 상위 계층에서 내려온 손실 함수의 그래디언트(∂L/∂y)입니다.
        let softmax_output = targets[0];
        let upstream_grad = grad;

        let s = softmax_output.data();
        let g = upstream_grad.data();

        // Softmax의 역전파 그래디언트(∂L/∂x_i)는 다음과 같이 계산됩니다:
        // ∂L/∂x_i = s_i * (g_i - dot(g, s))
        // 여기서 dot(g, s)는 상위 그래디언트와 softmax 출력의 내적입니다.

        // 1. 상위 그래디언트(g)와 Softmax 출력(s)의 내적(dot product)을 계산합니다.
        let dot_product: f32 = s.iter().zip(g.iter()).map(|(&s_val, &g_val)| s_val * g_val).sum();

        // 2. 각 입력에 대한 그래디언트를 계산합니다.
        let mut input_grad_data = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let grad_i = s[i] * (g[i] - dot_product);
            input_grad_data.push(grad_i);
        }

        // 최종 계산된 그래디언트를 텐서로 변환하여 반환합니다.
        Ok(vec![GlobalTensor::from_vec(input_grad_data, softmax_output.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    
    fn node_id(&self) -> &NodeId { &self.node_id }
}