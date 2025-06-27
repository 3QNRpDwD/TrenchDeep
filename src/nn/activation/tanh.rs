use crate::tensor::TENSOR_ALLOCATOR;
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
        let var_act = var_act!(output.to_id(false)?, self.label());
        var_act.with_grad_fn(self.operator.name(), &[&input]);
        Ok(var_act)
    }
    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        TENSOR_ALLOCATOR.with_borrow( |alloc| {
            let output = self.operator.forward(&[input])?.remove(0);
            Ok(alloc.get_tensor_ref(&output.id()).unwrap().clone())
        })
    }
    fn params(&self) -> Vec<&dyn Parameter> { vec![] }
    fn label(&self) -> &str { &self.label }
}

impl Function for Tanh {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Tanh)
    }

    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let x = targets[0];
        let pos_exp = self.backend.exp(&x.data());
        let neg_exp = self.backend.exp(&x.data().iter().map(|&val| -val).collect::<Vec<f32>>());

        // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Ok(vec![
            PooledTensor::from_vec(
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
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let tanh_output = targets[0];
        let ones = vec![1.0f32; tanh_output.data().len()];

        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * (1 - tanh^2(x))
        Ok(vec![
            PooledTensor::from_vec(
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