use super::*;

impl TanhLayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            cache: HashMap::new(),
            operator: Tanh::new().unwrap(),
        }
    }
}

impl Layer for TanhLayer {
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
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> { Ok(self.operator.forward(&[input])?.remove(0)) }
    fn params(&self) -> Vec<&dyn Parameter> { vec![] }
    fn inputs_cache(&self) -> &HashMap<HandleId, HandleId> { &self.cache }
    fn inputs_cache_mut(&mut self) -> &mut HashMap<HandleId, HandleId> { &mut self.cache }
    fn label(&self) -> &str { &self.label }
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

    #[cfg(all(feature = "enableBackpropagation"))]
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
    
    fn node_id(&self) -> &HandleId { &self.node_id }
}