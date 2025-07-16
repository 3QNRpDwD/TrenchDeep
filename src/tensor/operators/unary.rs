use super::*;

impl Function for Abs {
    /// Computes the absolute value of each element in the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute values of each element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(targets[0].data().iter().map(|&x| x.abs()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = targets[0];
        let sign_data: Vec<f32> = input.data().iter().map(|&x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }).collect();
        let sign_tensor = PooledTensor::from_vec(sign_data, input.shape())?;
        Ok(vec![grad * &sign_tensor])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl Function for Exp {
    /// Applies the exponential function to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being e ^ tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(self.backend().exp(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let gradiant = grad.data().iter()
            .zip(targets[0].data().iter())
            .map(|(grad_data, target_data)|  target_data.exp() * grad_data)
            .collect();

        Ok(vec![PooledTensor::from_vec(gradiant, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl Function for Log {
    /// Applies the natural logarithm to each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the natural logarithm of tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(targets[0].data().iter().map(|&x| x.ln()).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = targets[0];
        let grad_data: Vec<f32> = grad.data().iter().zip(input.data().iter()).map(|(g, i)| g / i).collect();
        Ok(vec![PooledTensor::from_vec(grad_data, input.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl Function for Pow {
    /// Raises each element in the tensor to a power
    ///
    /// # Arguments
    /// * `power` - The power to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element being tensor_element ^ power
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(self.backend().pow(targets[0].data(), self.power.unwrap()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let power = self.power.unwrap();
        let target = targets[0];
        let forwarded = PooledTensor::from_vec(self.backend().pow(target.data(), power - 1.0), target.shape())?; // x ** (c - 1)
        let result = PooledTensor::from_vec(
            forwarded
                .data()
                .iter()
                .map(|&x| power * x)
                .collect(), target.shape())?; // c * x ** (c - 1)
        Ok(vec![&result as &dyn TensorBase * grad]) // c * x ** (c -1) * gy
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl Function for Square {
    /// Returns a new tensor with the square of the elements of input
    ///
    /// # Returns
    /// A new tensor with each element being the square of the corresponding element in the input tensor
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(targets[0].data().iter().map(|x| x * x).collect(), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
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

        Ok(vec![PooledTensor::from_vec(gradient, targets[0].shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl Function for Sqrt {
    /// Takes the square root of each element in the tensor
    ///
    /// # Returns
    /// A new tensor with each element being the square root of tensor_element
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![PooledTensor::from_vec(self.backend().sqrt(targets[0].data()), targets[0].shape())?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = targets[0];
        let output = self.forward(targets)?;
        let grad_data: Vec<f32> = grad.data().iter().zip(output[0].data().iter()).map(|(g, o)| g / (2.0 * o)).collect();
        Ok(vec![PooledTensor::from_vec(grad_data, input.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::{MlResult, tensor::{TensorBase, Tensor}, variable};
    use crate::nn::Parameter;
    use crate::tensor::AutogradFunction;
    use crate::tensor::operators::{Abs, Function, Exp, Log, Pow, Square, Sqrt};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn tensor_abs_operator() -> MlResult<()> {
        let tensor = Tensor::new(vec![vec![-1.0, 2.0, -3.0]]);
        let op = Abs::new();
        let result = op.forward(&[&tensor])?.remove(0);
        assert_eq!(result.data(), vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_abs_backward() -> MlResult<()> {
        let a = variable!(vec![vec![-1.0, 2.0], vec![-3.0, 4.0]]);
        let op = Abs::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![-1.0, 1.0, -1.0, 1.0], &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }

    #[test]
    fn test_exp_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let op = Exp::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad_data: Vec<f32> = a.tensor().data().iter().map(|x| x.exp()).collect();
        let expected_grad = Tensor::from_vec(expected_grad_data, &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }

    #[test]
    fn test_log_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let op = Log::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad_data: Vec<f32> = a.tensor().data().iter().map(|x| 1.0 / x).collect();
        let expected_grad = Tensor::from_vec(expected_grad_data, &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }

    #[test]
    fn test_pow_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let op = Pow::new(Some(3.0));
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad_data: Vec<f32> = a.tensor().data().iter().map(|x| 3.0 * x.powi(2)).collect();
        let expected_grad = Tensor::from_vec(expected_grad_data, &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }

    #[test]
    fn test_square_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let op = Square::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad_data: Vec<f32> = a.tensor().data().iter().map(|x| 2.0 * x).collect();
        let expected_grad = Tensor::from_vec(expected_grad_data, &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }

    #[test]
    fn test_sqrt_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 4.0], vec![9.0, 16.0]]);
        let op = Sqrt::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad_data: Vec<f32> = a.tensor().data().iter().map(|x| 0.5 / x.sqrt()).collect();
        let expected_grad = Tensor::from_vec(expected_grad_data, &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}
