use super::*;


impl Function<f32> for Matmax {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), matmax: None }) }
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
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let buffer = match self.matmax.unwrap().0 {
            None => {
                // Find global maximum
                let max_val = targets[0].data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                vec![Tensor::<f32>::from_vec(vec![max_val], &vec![1])?, Tensor::<f32>::zeros()]
            }
            Some(d) => {
                let dim = if d < 0 {
                    (targets[0].shape().len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if dim >= targets[0].shape().len() {
                    return Err(MlError::TensorError(TensorError::InvalidAxis {
                        axis: dim,
                        shape: targets[0].shape().to_vec(),
                    }));
                }

                let mut new_shape = targets[0].shape().to_vec();
                if !self.matmax.unwrap().1 {
                    new_shape.remove(dim);
                } else {
                    new_shape[dim] = 1;
                }

                let stride: usize = targets[0].shape()[dim + 1..].iter().product();
                let outer_stride: usize = targets[0].shape()[dim..].iter().product();
                let outer_dims: usize = targets[0].shape()[..dim].iter().product();
                let dim_size = targets[0].shape()[dim];

                let mut max_values = Vec::with_capacity(targets[0].data().len() / dim_size);
                let mut max_indices = Vec::with_capacity(targets[0].data().len() / dim_size);

                for i in 0..outer_dims {
                    for j in 0..stride {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for k in 0..dim_size {
                            let idx = i * outer_stride + k * stride + j;
                            let val = targets[0].data()[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = k;
                            }
                        }

                        max_values.push(max_val);
                        max_indices.push(max_idx as f32);
                    }
                }

                vec![Tensor::<f32>::from_vec(max_values, &new_shape)?, Tensor::<f32>::from_vec(max_indices, &new_shape)?]
            }
        };

        Ok(buffer)
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}

#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Matmax};
    use crate::tensor::{Tensor, TensorBase};
    use crate::{tensor_ops, MlResult};
    #[test]
    fn test_max() -> MlResult<()> {
        // Test global maximum
        let buffer = Tensor::<f32>::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let (max_all, _) = tensor_ops!(buffer, Matmax, None, false);
        assert_eq!(max_all.data(), &[6.0]);

        // Test maximum along dimension 0
        let (max_dim0, indices0) = tensor_ops!(buffer, Matmax, Some(0), true);
        assert_eq!(max_dim0.shape(), &[1, 3]);
        assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(indices0.data(), &[1.0, 1.0, 1.0]);

        // Test maximum along dimension 1
        let (max_dim1, indices1) = tensor_ops!(buffer, Matmax, Some(1), true);
        assert_eq!(max_dim1.shape(), &[2, 1]);
        assert_eq!(max_dim1.data(), &[3.0, 6.0]);
        assert_eq!(indices1.data(), &[2.0, 2.0]);

        // Test maximum with negative dimension
        let (max_neg, indices_neg) = tensor_ops!(buffer, Matmax, Some(-1), true);
        assert_eq!(max_neg.data(), &[3.0, 6.0]);
        assert_eq!(indices_neg.data(), &[2.0, 2.0]);

        Ok(())
    }
}