use super::*;

impl Function for Matmax {
    /// Returns the maximum value of all elements in the input tensor.
    /// If dim is specified, returns the maximum values along the given dimension.
    ///
    /// # Arguments
    /// * `dim` - Optional dimension along which to find the maximum values
    /// * `keepdim` - Whether the output tensor has dim retained or not
    ///
    /// # Returns
    /// If dim is None, returns a tensor with a single element containing the maximum value.
    /// If dim is specified, returns a tuple of two tensors (values, indices) containing the
    /// maximum values and their indices along the specified dimension.
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let target_0 = targets[0];
        let target_0_shape = target_0.shape();
        let target_0_data = target_0.data();
        let buffer = match self.matmax.unwrap().0 {
            None => {
                // Find global maximum
                let max_val = target_0_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                vec![PooledTensor::from_vec(vec![max_val], &vec![1])?, PooledTensor::zeros(target_0_shape)]
            }
            Some(d) => {
                let dim = if d < 0 {
                    (target_0_shape.len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= target_0_shape.len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: target_0_shape.to_vec(),
                    }));
                }

                let mut new_shape = target_0_shape.to_vec();
                if !self.matmax.unwrap().1 {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = target_0_shape[dim + 1..].iter().product();
                let outer_stride: usize = target_0_shape[dim..].iter().product();
                let outer_dims: usize = target_0_shape[..dim].iter().product();
                let dim_size = target_0_shape[dim];

                let mut max_values = Vec::with_capacity(target_0_data.len() / dim_size);
                let mut max_indices = Vec::with_capacity(target_0_data.len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = target_0_data[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = k;
                            }
                        }

                        max_values.push(max_val);
                        max_indices.push(max_idx as f32);
                    }
                }

                vec![PooledTensor::from_vec(max_values, &new_shape)?, PooledTensor::from_vec(max_indices, &new_shape)?]
            }
        };

        Ok(buffer)
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let input = targets[0];
        let output = self.forward(targets)?;
        let max_indices = &output[1];

        let mut grad_input_data = vec![0.0; input.data().len()];

        match self.matmax.unwrap().0 {
            None => {
                let max_val = output[0].data()[0];
                if let Some(pos) = input.data().iter().position(|&r| r == max_val) {
                    grad_input_data[pos] = grad.data()[0];
                }
            }
            Some(d) => {
                let dim = if d < 0 {
                    (input.shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                let stride: usize = input.shape()[dim + 1..].iter().product();
                let outer_stride: usize = input.shape()[dim..].iter().product();
                let outer_dims: usize = input.shape()[..dim].iter().product();

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let grad_idx = i * stride + j;
                        let max_idx = max_indices.data()[grad_idx] as usize;
                        let input_idx = i * outer_stride + max_idx * stride + j;
                        grad_input_data[input_idx] = grad.data()[grad_idx];
                    }
                }
            }
        }

        Ok(vec![PooledTensor::from_vec(grad_input_data, input.shape())?])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Matmax};
    use crate::{tensor_ops, MlResult, variable};
    use crate::nn::Parameter;
    use crate::tensor::{AutogradFunction, Tensor, TensorBase};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn test_max() -> MlResult<()> {
        let buffer = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let result = tensor_ops!(buffer, Matmax, Matmax = Some((None, false)));
        let max_all = &result.0;
        assert_eq!(max_all.data(), &[6.0]);

        // Test maximum along dimension 0
        let result = tensor_ops!(buffer, Matmax, Matmax = Some((Some(0), true)));
        let max_dim0 = &result.0;
        let indices0 = &result.1;
        assert_eq!(max_dim0.shape(), &[1, 3]);
        assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(indices0.data(), &[1.0, 1.0, 1.0]);

        // Test maximum along dimension 1
        let result = tensor_ops!(buffer, Matmax, Matmax = Some((Some(1), true)));
        let max_dim1 = &result.0;
        let indices1 = &result.1;
        assert_eq!(max_dim1.shape(), &[2, 1]);
        assert_eq!(max_dim1.data(), &[3.0, 6.0]);
        assert_eq!(indices1.data(), &[2.0, 2.0]);

        // Test maximum with negative dimension
        let result =  tensor_ops!(buffer, Matmax, Matmax = Some((Some(-1), true)));
        let max_neg = &result.0;
        let indices_neg = &result.1;
        assert_eq!(max_neg.data(), &[3.0, 6.0]);
        assert_eq!(indices_neg.data(), &[2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_matmax_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 5.0, 2.0], vec![4.0, 3.0, 6.0]]);
        let op = Matmax::new(Some((Some(1), true)));
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}
