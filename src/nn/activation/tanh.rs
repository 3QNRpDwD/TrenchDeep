use super::*;

impl TanhLayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            inputs: HashSet::new(),
            outputs: HashMap::new(),
            operator: Tanh::new().unwrap(),
        }
    }
}

impl Layer for TanhLayer {
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

impl Function for Tanh {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Tanh)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let x = targets[0];
        let pos_exp = self.backend.exp(&x.data());
        let neg_exp = self.backend.exp(&x.data().iter().map(|&val| -val).collect::<Vec<f32>>());

        // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.div(
                    &self.backend.sub(
                        &pos_exp,
                        &neg_exp
                    ),
                    &self.backend.add(
                        &pos_exp,
                        &neg_exp
                    )
                ),
                x.shape()
            )?
        ])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let tanh_output = targets[0];
        let ones = vec![1.0f32; tanh_output.data().len()];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * (1 - tanh^2(x))
        Ok(vec![
            GlobalTensor::from_vec(
                self.backend.multiply(
                    &grad.data(),
                    &self.backend.sub(
                        &ones,
                        &self.backend.multiply(
                            &tanh_output.data(),
                            &tanh_output.data()
                        )
                    )
                ),
                grad.shape()
            )?
        ])
    }
    
    fn node_id(&self) -> &NodeId { &self.node_id }
}