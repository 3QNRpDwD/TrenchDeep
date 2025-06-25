use super::*;

impl ReLULayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            cache: HashMap::new(),
            operator: ReLU::new().unwrap()
        }
    }
}

impl Layer for ReLULayer {
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
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> { Ok(self.operator.forward(&[input])?.remove(0))}
    fn params(&self) -> Vec<&dyn Parameter> { vec![] }
    fn label(&self) -> &str { &self.label }
}

impl Function for ReLU {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(ReLU)
    }

    fn forward(&self, x: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        // ReLU(x) = max(0, x)
        let result = x[0].data().iter()
            .map(|&val| if val > 0.0 { val } else { 0.0 })
            .collect::<Vec<f32>>();

        Ok(vec![GlobalTensor::from_vec(result, x[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, target: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let relu_output = target[0];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * mask
        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &relu_output.data().iter()
                        .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
                        .collect::<Vec<f32>>()
                ),
                grad.shape()
            )?
        ])
    }
    
    fn node_id(&self) -> &HandleId { &self.node_id }
}