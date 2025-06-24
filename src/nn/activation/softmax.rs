use super::*;

impl SoftmaxLayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            cache: HashMap::new(),
            operator: Softmax::new().unwrap(),
        }
    }
}

impl Layer for SoftmaxLayer {
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let output = self.operator.forward(&[input.tensor()])?.remove(0);
        let in_id = input.node_id();
        let applied = match self.cache.contains_key(&in_id) {
            true => output.with_id(*self.cache.get(&in_id).unwrap())?,
            false => {
                let temp = output.to_id()?;
                self.cache.insert(in_id, temp.id());
                temp
            }
        };
        let var_act = var_act!(applied, self.label());
        var_act.with_grad_fn(self.operator.name(), &[&input]);
        Ok(var_act)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        Ok(self.operator.forward(&[input])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }
    fn inputs_cache(&self) -> &HashMap<NodeId, NodeId> { &self.cache }
    fn inputs_cache_mut(&mut self) -> &mut HashMap<NodeId, NodeId> { &mut self.cache }
    fn label(&self) -> &str { &self.label }
}

impl Function for Softmax {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Softmax)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input = targets[0];
        let input_data = input.data();
        let max_val = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = input_data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_of_exps: f32 = exp_values.iter().sum();
        let softmax_output: Vec<f32> = exp_values.iter().map(|&exp_val| exp_val / sum_of_exps).collect();

        Ok(vec![GlobalTensor::from_vec(softmax_output, input.shape())?])
    }

    /// Softmax 함수의 역전파(gradient)를 계산합니다.
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let softmax_output = targets[0];
        let upstream_grad = grad;
        let s = softmax_output.data();
        let g = upstream_grad.data();
        // ∂L/∂x_i = s_i * (g_i - dot(g, s))
        let dot_product: f32 = s.iter().zip(g.iter()).map(|(&s_val, &g_val)| s_val * g_val).sum();
        let mut input_grad_data = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let grad_i = s[i] * (g[i] - dot_product);
            input_grad_data.push(grad_i);
        }

        Ok(vec![GlobalTensor::from_vec(input_grad_data, softmax_output.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    
    fn node_id(&self) -> &NodeId { &self.node_id }
}