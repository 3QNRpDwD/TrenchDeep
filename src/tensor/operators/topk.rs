use super::*;

impl Function for Topk {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "Topk";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Box::new(Topk { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next() })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }
    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `targets[0]` - Input tensor
    /// * `targets[1]` - k (Scalar Tensor)
    /// * `targets[2]` - sorted (Scalar Tensor, > 0.5 is true) (Optional, default true)
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let input = targets[0];
        let k_val = if targets.len() > 1 {
            targets[1].data()[0] as usize
        } else {
             return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be provided".to_string(),
            }));
        };
        
        let sorted = if targets.len() > 2 {
            targets[2].data()[0] > 0.5
        } else {
            true
        };

        if k_val == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = input.shape().len() - 1;
        let last_dim_size = input.shape()[last_dim];

        if k_val > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    k_val, last_dim_size
                ),
            }));
        }

        let slice_size = last_dim_size;
        let num_slices: usize = input.shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * k_val);
        let mut indices = Vec::with_capacity(num_slices * k_val);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &input.data()[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..k_val];
            let mut selected = top_k.to_vec();
            if !sorted {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = input.shape().to_vec();
        new_shape[last_dim] = k_val;

        Ok(vec![GlobalTensor::from_vec(values, &new_shape)?, GlobalTensor::from_vec(indices, &new_shape)?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}


#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Topk};
    use crate::tensor::{Tensor, TensorBase};
    use crate::{tensor_ops, MlResult};

    #[test]
    fn test_topk() -> MlResult<()> {
        // Test 1: Basic 1D tensor
        let buffer = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let k = Tensor::scalar(3.0);
        let sorted = Tensor::scalar(1.0); // true
        let mut result = Topk::new().unwrap().forward(&[&buffer, &k, &sorted])?;
        let values = result.remove(0);
        let indices = result.remove(0);
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let buffer = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0], &[2, 5], )?;
        let (values, indices) = tensor_ops!(buffer, Topk, 2, true);
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let buffer = Tensor::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor_ops!(buffer, Topk ,3, false);
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
}
