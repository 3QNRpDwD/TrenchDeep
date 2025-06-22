use super::*;

impl Function for Abs {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Abs)
    }
    
    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| x.abs()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl Function for Exp {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Exp)
    }
    
    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(self.backend().exp(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)|  target_data.exp() * grad_data)
            .collect();

        Ok(vec![GlobalTensor::from_vec(gradiant, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl Function for Log {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Log)
    }
    
    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|&x| x.ln()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl Function for Pow {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "Pow";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Box::new(Pow { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), power: None })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }
    
    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(self.backend().pow(targets[0].data(), self.power.unwrap()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let power = self.power.unwrap();
        let target = targets[0];
        let forwarded = GlobalTensor::from_vec(self.backend().pow(target.data(), power - 1.0), target.shape())?; // x ** (c - 1)
        let result = GlobalTensor::from_vec(
            forwarded
                .data()
                .iter()
                .map(|&x| power * x)
                .collect(), target.shape())?; // c * x ** (c - 1)
        Ok(vec![&result as &dyn TensorBase * grad]) // c * x ** (c -1) * gy
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl Function for Square {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Square)
    }
    
    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(targets[0].data().iter().map(|x| x * x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        // grad가 scalar이거나 다른 shape일 때 브로드캐스팅
        let grad_broadcasted = if grad.data().len() == 1 {
            // grad가 scalar인 경우, targets[0]와 같은 길이로 복제
            vec![grad.data()[0]; targets[0].data().len()]
        } else {
            grad.data().to_vec()
        };

        let gradient: Vec<f32> = grad_broadcasted.iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)| grad_data * 2.0 * target_data)
            .collect();

        Ok(vec![GlobalTensor::from_vec(gradient, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}

impl Function for Sqrt {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sqrt)
    }
    
    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        Ok(vec![GlobalTensor::from_vec(self.backend().sqrt(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}