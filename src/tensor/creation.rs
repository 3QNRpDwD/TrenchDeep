use crate::{MlError, MlResult};
use crate::tensor::{TensorBase, Tensor, TensorError, ArcTensor};


impl  Tensor<f32> {
    pub fn zeros() -> ArcTensor<f32> {
        ArcTensor::new(Self {
            data: vec![],
            shape: vec![],
            requires_grad: cfg!(feature = "enable_backpropagation"),

            #[cfg(feature = "enable_backpropagation")]
            grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        })
    }

    pub fn scalar(scalar: f32) -> ArcTensor<f32> {
        ArcTensor::new(Self {
            data: vec![scalar],
            shape: vec![1],
            requires_grad: cfg!(feature = "enable_backpropagation"),

            #[cfg(feature = "enable_backpropagation")]
            grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        })
    }
}

impl TensorBase<f32> for Tensor<f32> {
    fn new(data: Vec<Vec<f32>>) -> ArcTensor<f32>  {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<f32> = data.into_iter().flatten().collect();

        ArcTensor::new(Self {
            data,
            shape,
            requires_grad: cfg!(feature = "enable_backpropagation"),

            #[cfg(feature = "enable_backpropagation")]
            grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        })
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> MlResult<ArcTensor<f32>> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(ArcTensor::new(Self {
            data,
            shape: shape.to_vec(),
            requires_grad: cfg!(feature = "enable_backpropagation"),

            #[cfg(feature = "enable_backpropagation")]
            grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        }))
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[f32] {
        &self.data
    }

    fn get(&self, indices: &[usize]) -> Option<&f32> {
        self.data.get(self.index(indices)?)
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        Some(
            indices
                .iter()
                .zip(&self.shape)
                .fold(0, |acc, (&i, &dim)| acc * dim + i),
        )
    }

    /// Verifies if two tensors can perform element-wise operations
    ///
    /// # Arguments
    /// * `other` - The tensor to compare shapes with
    ///
    /// # Returns
    /// * `Ok(())` if the shapes match
    /// * `Err(MlError::TensorError)` if shapes don't match
    fn chk_shape(&self, other: &dyn TensorBase<f32>) -> MlResult<()> {
        if self.shape != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape.to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    // #[cfg(feature = "enable_backpropagation")]
    // fn set_grad_fn(&mut self, grad_fn: Box<dyn crate::tensor::Function<'static, f32, Forwarded=(), Gradiant=()>>) {
    //     self.grad_fn = Some(grad_fn);
    // }
    //
    // #[cfg(feature = "enable_backpropagation")]
    // fn grad(&self) -> Option<&dyn TensorBase<f32>> {
    //     self.grad.as_ref().map(|g| g.as_ref())
    // }
}