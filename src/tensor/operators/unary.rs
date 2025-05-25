use super::*;

impl Function for Abs {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(targets[0].data().iter().map(|&x| x.abs()).collect(), targets[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function for Exp {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(self.backend().exp(targets[0].data().as_slice()), targets[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)|  target_data.exp() * grad_data)
            .collect();

        Ok(vec![Tensor::from_vec(gradiant, targets[0].shape().as_slice())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function for Log {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(targets[0].data().iter().map(|&x| x.ln()).collect(), targets[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function for Pow {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), power: None }) }
    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(self.backend().pow(targets[0].data().as_slice(), self.power.unwrap()), targets[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let power = self.power.unwrap();
        let target = targets[0];
        let forwarded = Tensor::from_vec(self.backend().pow(target.data().as_slice(), power - 1.0), target.shape().as_slice())?; // x ** (c - 1)
        let result = Tensor::from_vec(
            forwarded
                .data()
                .iter()
                .map(|&x| power * x)
                .collect(), target.shape().as_slice())?; // c * x ** (c - 1)
        Ok(vec![result * grad]) // c * x ** (c -1) * gy
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function for Square {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(targets[0].data().iter().map(|x| x * x).collect(), targets[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)| 2.0  * target_data * grad_data )
            .collect();

        Ok(vec![Tensor::from_vec(gradiant, targets[0].shape().as_slice())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function for Sqrt {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self, targets: &[Tensor]) -> MlResult<Vec<Tensor>> {
        Ok(vec![Tensor::from_vec(self.backend().sqrt(targets[0].data().as_slice()), targets[0].shape().as_slice())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[Tensor], grad: Tensor) -> MlResult<Vec<Tensor>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}