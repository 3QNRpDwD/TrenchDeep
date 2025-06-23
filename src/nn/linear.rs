use crate::tensor::AutogradFunction;
use super::*;

impl Linear {
    /// 새로운 Linear 레이어를 생성합니다.
    ///
    /// # Arguments
    /// * `in_features` - 입력 텐서의 특성(피처) 수
    /// * `out_features` - 출력 텐서의 특성(피처) 수
    /// * `label` - 시각화 및 디버깅을 위한 레이어의 이름
    pub fn new(in_features: usize, out_features: usize, label: &str) -> MlResult<Self> {
        // 가중치(W) 초기화: (in_features, out_features) 형태
        // Kaiming He 초기화와 유사하게 표준편차를 조절하여 가중치를 초기화합니다.
        // 이는 학습 초기 단계에서 그래디언트가 소실되거나 폭발하는 것을 방지하는 데 도움이 됩니다.
        let k = 1.0 / (in_features as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rand::random::<f32>() * 2.0 * k - k)
            .collect();
        let weight_tensor = Tensor::from_vec(weight_data, &[in_features, out_features])?;

        // 편향(b) 초기화: (out_features) 형태
        let bias_data = vec![0.0; out_features];
        let bias_tensor = Tensor::from_vec(bias_data, &[out_features])?;

        Ok(Self {
            label: label.to_string(),
            weight: var_weight!(weight_tensor),
            bias: var_bias!(bias_tensor),
            inputs: HashSet::new(),
            outputs: HashMap::new(),
        })
    }
}


impl Layer for Linear {
    fn apply(&mut self, input: Arc<Variable>) -> MlResult<Arc<Variable>> {
        let mut matmul = Matmul::new()?;
        let mut add = Add::new()?;

        // 2. y = xW + b 연산을 수행합니다.
        let x = matmul.apply(&[&input, &self.weight])?;
        let output = add.apply(&[&x, &self.bias])?;

        // 3. 입/출력 노드 ID를 캐시에 저장합니다.
        self.inputs.insert(input.node_id());
        self.outputs.insert(input.node_id(), output.node_id());

        Ok(output)
    }

    /// 계산 그래프를 구성하지 않고 순수하게 예측만 수행합니다. (추론 시 사용)
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        // 1. 사용할 연산자의 핸들을 가져옵니다.
        let matmul = Matmul::new()?;
        let add = Add::new()?;

        // 2. Variable에서 실제 Tensor 데이터를 가져옵니다.
        let weight_tensor = self.weight.tensor();
        let bias_tensor = self.bias.tensor();

        // 3. Function 트레이트의 forward 메소드를 직접 호출하여 순수 계산만 수행합니다.
        let x = matmul.forward(&[input, weight_tensor])?.remove(0);
        let output = add.forward(&[&x, bias_tensor])?.remove(0);

        Ok(output)
    }

    /// 이 레이어가 소유한 모든 파라미터(가중치, 편향)의 참조를 반환합니다.
    fn params(&self) -> Vec<&dyn Parameter> {
        vec![self.weight.as_ref(), self.bias.as_ref()]
    }

    fn inputs_cache(&self) -> &HashSet<NodeId> {
        &self.inputs
    }

    fn outputs_cache(&self) -> &HashMap<NodeId, NodeId> {
        &self.outputs
    }


    fn inputs_cache_mut(&mut self) -> &mut HashSet<NodeId> {
        &mut self.inputs
    }

    fn outputs_cache_mut(&mut self) -> &mut HashMap<NodeId, NodeId> {
        &mut self.outputs
    }

    fn label(&self) -> &str {
        &self.label
    }
}