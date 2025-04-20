use super::*;

impl Function<f32> for Abs {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| x.abs()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Exp {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(self.backend().exp(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)|  target_data.exp() * grad_data)
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradiant, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Log {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|&x| x.ln()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Pow {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), power: None }) }
    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(self.backend().pow(targets[0].data(), self.power.unwrap()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let power = self.power.unwrap();
        let target = targets[0];
        let forwarded = Tensor::<f32>::from_vec(self.backend().pow(target.data(), power - 1.0), target.shape())?; // x ** (c - 1)
        let result = Tensor::from_vec(
            forwarded
                .data()
                .iter()
                .map(|&x| power * x)
                .collect(), target.shape())?; // c * x ** (c - 1)
        Ok(vec![result * grad]) // c * x ** (c -1) * gy
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Square {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(targets[0].data().iter().map(|x| x * x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)| 2.0  * target_data * grad_data )
            .collect();

        Ok(vec![Tensor::<f32>::from_vec(gradiant, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

impl Function<f32> for Sqrt {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?) }) }
    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![Tensor::<f32>::from_vec(self.backend().sqrt(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}