use super::*;

impl Function<f32> for Topk {
    fn new() -> MlResult<Self> { Ok(Self { backend: Arc::new(CpuBackend::new()?), topk: None }) }
    /// Returns the k largest elements of the tensor along the last dimension.
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `sorted` - Whether to return the elements in sorted order
    ///
    /// # Returns
    /// A tuple of two tensors (values, indices) containing the top k values and their indices
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        if self.topk.unwrap().0 == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: "k must be greater than 0".to_string(),
            }));
        }

        let last_dim = targets[0].shape().len() - 1;
        let last_dim_size = targets[0].shape()[last_dim];

        if self.topk.unwrap().0 > last_dim_size {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "topk",
                reason: format!(
                    "k ({}) cannot be larger than last dimension size ({})",
                    self.topk.unwrap().0, last_dim_size
                ),
            }));
        }

        let slice_size = last_dim_size;
        let num_slices: usize = targets[0].shape()[..last_dim].iter().product();
        let mut values = Vec::with_capacity(num_slices * self.topk.unwrap().0);
        let mut indices = Vec::with_capacity(num_slices * self.topk.unwrap().0);


        for slice_idx in 0..num_slices {
            let start_idx = slice_idx * slice_size;
            let end_idx = start_idx + slice_size;
            let slice_data = &targets[0].data()[start_idx..end_idx];
            let mut pairs: Vec<(f32, usize)> = slice_data
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect();


            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));


            let top_k = &pairs[..self.topk.unwrap().0];
            let mut selected = top_k.to_vec();
            if !self.topk.unwrap().1 {
                selected.sort_by_key(|pair| pair.1);
            }

            values.extend(selected.iter().map(|pair| pair.0));
            indices.extend(selected.iter().map(|pair| pair.1 as f32));
        }

        let mut new_shape = targets[0].shape().to_vec();
        new_shape[last_dim] = self.topk.unwrap().0;

        Ok(vec![Tensor::<f32>::from_vec(values, &new_shape)?, Tensor::<f32>::from_vec(indices, &new_shape)?])
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        todo!()
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
}


#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Function, Topk};
    use crate::tensor::{Tensor, TensorBase};
    use crate::{tensor_ops, MlResult};

    #[test]
    fn test_topk() -> MlResult<()> {
        // Test 1: Basic 1D tensor
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor_ops!(buffer, Topk, 3, true);
        assert_eq!(values.data(), &[5.0, 4.0, 3.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 2.0]);

        // Test 2: 2D tensor
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0, 2.0, 3.0, 1.0, 4.0, 5.0], &[2, 5], )?;
        let (values, indices) = tensor_ops!(buffer, Topk, 2, true);
        assert_eq!(values.shape(), &[2, 2]);
        assert_eq!(values.data(), &[5.0, 4.0, 5.0, 4.0]);
        assert_eq!(indices.data(), &[4.0, 1.0, 4.0, 3.0]);

        // Test 3: Unsorted output
        let buffer = Tensor::<f32>::from_vec(vec![1.0, 4.0, 3.0, 2.0, 5.0], &[5])?;
        let (values, indices) = tensor_ops!(buffer, Topk ,3, false);
        assert_eq!(values.data(), &[4.0, 3.0, 5.0]);
        assert_eq!(indices.data(), &[1.0, 2.0, 4.0]);

        Ok(())
    }
}