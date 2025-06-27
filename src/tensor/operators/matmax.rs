use super::*;

impl Function for Matmax {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "Matmax";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Box::new(Matmax { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), matmax: None })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }
    
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
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::MlResult;

    #[test]
    fn test_max() -> MlResult<()> {
        // Test global maximum
        // let buffer = Tensor::<f32>::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        // let (max_all, _) = tensor_ops!(buffer, Matmax, None, false);
        // assert_eq!(max_all.data(), &[6.0]);
        // 
        // // Test maximum along dimension 0
        // let (max_dim0, indices0) = tensor_ops!(buffer, Matmax, Some(0), true);
        // assert_eq!(max_dim0.shape(), &[1, 3]);
        // assert_eq!(max_dim0.data(), &[4.0, 5.0, 6.0]);
        // assert_eq!(indices0.data(), &[1.0, 1.0, 1.0]);
        // 
        // // Test maximum along dimension 1
        // let (max_dim1, indices1) = tensor_ops!(buffer, Matmax, Some(1), true);
        // assert_eq!(max_dim1.shape(), &[2, 1]);
        // assert_eq!(max_dim1.data(), &[3.0, 6.0]);
        // assert_eq!(indices1.data(), &[2.0, 2.0]);
        // 
        // // Test maximum with negative dimension
        // let (max_neg, indices_neg) = tensor_ops!(buffer, Matmax, Some(-1), true);
        // assert_eq!(max_neg.data(), &[3.0, 6.0]);
        // assert_eq!(indices_neg.data(), &[2.0, 2.0]);
        
        todo!("Implement tests for Matmax operator"); // Placeholder for actual test implementation

        Ok(())
    }
}