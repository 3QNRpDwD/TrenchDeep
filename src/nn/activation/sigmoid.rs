use super::*;

impl SigmoidLayer {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            operator: Sigmoid::new().unwrap(),
        }
    }
}

impl Layer for SigmoidLayer {
    #[cfg(feature = "enableBackpropagation")]
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

impl Function for Sigmoid {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sigmoid)
    }
    
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let x = targets[0];
        let ones = vec![1.0f32; x.data().len()];
        Ok(vec![
            PooledTensor::from_vec(
                self.backend.div(&ones, &self.backend.add(&ones, &self.backend.exp(x.data()))),
                x.shape()
            )?]
        )
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input_x = targets[0];
        // σ'(x) = σ(x) * (1 - σ(x))
        // ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * σ'(x)

        //시그모이드 출력값 σ(x)
        let neg_x_data = input_x.data().iter().map(|&v| -v).collect::<Vec<f32>>();
        let ones = vec![1.0f32; input_x.data().len()];
        let exp_neg_x = self.backend.exp(&neg_x_data);
        let one_plus_exp = self.backend.add(&ones, &exp_neg_x);
        let sigmoid_output_data = self.backend.div(&ones, &one_plus_exp); // σ(x)

        // derivative = σ(x) * (1 - σ(x))
        let derivative = self.backend.multiply(
            &sigmoid_output_data,
            &self.backend.sub(
                &ones,
                &sigmoid_output_data
            )
        );

        // 최종 그래디언트: ∂L/∂x = ∂L/∂y * ∂y/∂x = grad * derivative
        Ok(vec![
            PooledTensor::from_vec(
                self.backend.multiply(&grad.data(), &derivative),
                grad.shape()
            )?
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
    
    fn node_id(&self) -> &HandleId { &self.node_id }
}