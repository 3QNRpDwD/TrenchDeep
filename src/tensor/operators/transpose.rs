use super::*;


impl Function<f32> for Transpose {
    fn new() -> MlResult<GlobalFunction> {
        OPERATOR_STORAGE.with(|ops| {
            let my = "Transpose";
            let mut ops = ops.borrow_mut();
            match ops.contains_key(my) {
                true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                false => {
                    ops.insert(
                        String::from(my),
                        Arc::new(Transpose { backend: Arc::new(CpuBackend::new()?), node_id: NODE_ID_GEN.next(), dims: (0, 0) })
                    );
                    Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                }
            }
        })
    }
    
    fn forward(&self, input: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
        let input = input[0];
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        // Convert negative dimensions to positive
        let d0 = if self.dims.0 < 0 { rank as i32 + self.dims.0 } else { self.dims.0 } as usize;
        let d1 = if self.dims.1 < 0 { rank as i32 + self.dims.1 } else { self.dims.1 } as usize;

        if d0 >= rank || d1 >= rank {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: d0.max(d1),
                shape: input.shape.clone(),
            }));
        }

        // Create new shape with dimensions swapped
        let mut new_shape = input.shape.clone();
        new_shape.swap(d0, d1);

        // Calculate strides for the original shape
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * input.shape[i + 1];
        }

        // Create transposed data
        let mut result = vec![0.0; input.data.len()];
        let mut coords = vec![0usize; rank];

        for i in 0..input.data.len() {
            // Calculate source coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Swap the specified dimensions
            coords.swap(d0, d1);

            // Calculate target index
            let mut target_idx = 0;
            let mut stride = 1;
            for j in (0..rank).rev() {
                target_idx += coords[j] * stride;
                stride *= new_shape[j];
            }

            result[target_idx] = input.data[i];
        }

        Ok(vec![Tensor::from_vec(result, &new_shape)?])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, _: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        let input = grad;
        let rank = input.shape().len();
        if rank < 2 {
            return Err(MlError::TensorError(TensorError::InvalidOperation {
                op: "transpose",
                reason: "Tensor must have at least 2 dimensions".to_string(),
            }));
        }

        // Convert negative dimensions to positive
        let d0 = if self.dims.0 < 0 { rank as i32 + self.dims.0 } else { self.dims.0 } as usize;
        let d1 = if self.dims.1 < 0 { rank as i32 + self.dims.1 } else { self.dims.1 } as usize;

        if d0 >= rank || d1 >= rank {
            return Err(MlError::TensorError(TensorError::InvalidAxis {
                axis: d0.max(d1),
                shape: input.shape.clone(),
            }));
        }

        // Create new shape with dimensions swapped
        let mut new_shape = input.shape.clone();
        new_shape.swap(d0, d1);

        // Calculate strides for the original shape
        let mut strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * input.shape[i + 1];
        }

        // Create transposed data
        let mut result = vec![0.0; input.data.len()];
        let mut coords = vec![0usize; rank];

        for i in 0..input.data.len() {
            // Calculate source coordinates
            let mut idx = i;
            for j in 0..rank {
                coords[j] = idx / strides[j];
                idx %= strides[j];
            }

            // Swap the specified dimensions
            coords.swap(d0, d1);

            // Calculate target index
            let mut target_idx = 0;
            let mut stride = 1;
            for j in (0..rank).rev() {
                target_idx += coords[j] * stride;
                stride *= new_shape[j];
            }

            result[target_idx] = input.data[i];
        }

        Ok(vec![Tensor::from_vec(result, &new_shape)?])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn node_id(&self) -> &NodeId { &self.node_id }
}