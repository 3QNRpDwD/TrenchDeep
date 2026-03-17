use super::*;
impl Conv {
    pub fn new(
        label: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> MlResult<Self> {
        let k = 1.0 / (in_channels as f32 * kernel_size.0 as f32 * kernel_size.1 as f32).sqrt();
        let weight_shape = vec![out_channels, in_channels, kernel_size.0, kernel_size.1];
        let weight_data: Vec<f32> = (0..weight_shape.iter().product())
            .map(|_| rand::random::<f32>() * 2.0 * k - k)
            .collect();
        let weight = var_weight!(Tensor::from_vec(weight_data, &weight_shape)?);

        let bias_var = if bias {
            let bias_shape = vec![out_channels];
            Some(var_bias!(Tensor::zeros(&bias_shape)))
        } else {
            None
        };

        Ok(Self {
            label: label.to_string(),
            weight,
            bias: bias_var,
            conv2d: Conv2d::new(kernel_size, stride, padding),
        })
    }
}

impl Layer for Conv {
    #[cfg(feature = "enableBackpropagation")]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let inputs_for_grad: Vec<&Variable> = if let Some(b) = &self.bias {
            vec![input, &self.weight, b]
        } else {
            vec![input, &self.weight]
        };

        let output_tensor = self.conv2d.forward(&[input.tensor()])?.remove(0);
        let output_variable = var_act!(output_tensor.to_id(true)?, self.label());
        let op: Arc<dyn Function + Send + Sync> = self.conv2d.clone();
        output_variable.with_grad_fn(op, &inputs_for_grad);

        Ok(output_variable)
    }

    fn predict(&mut self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let inputs: Vec<&dyn TensorBase> = if let Some(b) = &self.bias {
            vec![input, self.weight.tensor(), b.tensor()]
        } else {
            vec![input, self.weight.tensor()]
        };
        
        let pooled_tensor = self.conv2d.forward(&inputs)?.remove(0);
        TENSOR_ALLOCATOR.with_borrow(|alloc| {
            Ok(alloc.get_tensor_ref(&pooled_tensor.id()).unwrap().clone())
        })
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        let mut params: Vec<&dyn Parameter> = vec![&self.weight];
        if let Some(b) = &self.bias {
            params.push(b);
        }
        params
    }

    fn label(&self) -> &str {
        &self.label
    }
}