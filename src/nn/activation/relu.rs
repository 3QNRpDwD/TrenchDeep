use super::*;

impl ReLULayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            inputs: HashSet::new(),
            outputs: HashMap::new(),
            operator: ReLU::new().unwrap()
        }
    }
}

impl Layer for ReLULayer {
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let output = self.operator.forward(&[input.tensor()])?.remove(0);
        let applied = match self.inputs.contains(&input.node_id()) {
            true => output.with_id(*self.outputs.get(&input.node_id()).unwrap())?,
            false => {
                let tensor = output.to_id()?;
                self.inputs.insert(input.node_id());
                tensor
            }
        };
        let var_act = var_act!(applied, self.label());
        var_act.with_grad_fn(self.operator.type_name(), &[&input]);
        Ok(var_act)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        Ok(self.operator.forward(&[input])?.remove(0))
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
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
    
    fn node_id(&self) -> &NodeId { &self.node_id }
}