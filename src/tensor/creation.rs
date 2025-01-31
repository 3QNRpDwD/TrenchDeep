use crate::{MlError, MlResult};
use crate::tensor::{TensorBase, Tensor, TensorError};


impl<T> TensorBase<T> for Tensor<T> {
    fn new(data: Vec<Vec<T>>) -> MlResult<Self> {
        let shape = vec![data.len(), data[0].len()];
        let data: Vec<T> = data.into_iter().flatten().collect();

        Ok(Self {
            data,
            shape,
            grad: None,
            grad_fn: None,
            requires_grad: false,
        })
    }

    fn from_vec(data: Vec<T>, shape: &[usize]) -> MlResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(MlError::TensorError(TensorError::InvalidDataLength {
                expected: expected_len,
                got: data.len(),
            }));
        }

        Ok(Self {
            data,
            shape: shape.to_vec(),
            grad: None,
            grad_fn: None,
            requires_grad: false,
        })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> &[T] {
        &self.data
    }

    fn get(&self, indices: &[usize]) -> Option<&T> {
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
    fn chk_shape(&self, other: &T) -> MlResult<()> {
        if self.shape() != other.shape() {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }));
        }
        Ok(())
    }
}

// impl<T: TensorBase Function<T> for Functions {
//     type Output = MlResult<T>;
//     type Gradient = f64;
//
//     fn forward(&self) -> Self::Output {
//         match self {
//             Functions::Abs      (F) => F.forward(),
//             Functions::Exp      (F) => F.forward(),
//             Functions::Log      (F) => F.forward(),
//             Functions::Neg      (F) => F.forward(),
//             Functions::Sqrt     (F) => F.forward(),
//             Functions::Square   (F) => F.forward(),
//
//             Functions::Add      (F) => F.forward(),
//             Functions::Sub      (F) => F.forward(),
//             Functions::Mul      (F) => F.forward(),
//             Functions::Div      (F) => F.forward(),
//             Functions::Pow      (F) => F.forward(),
//             Functions::Matmul   (F) => F.forward(),
//
//             Functions::Topk     (F) => F.forward(),
//             Functions::Matmax   (F) => F.forward(),
//         }
//     }
//     fn backward(&self, grad: Self::Gradient) -> Self::Output {
//         match self {
//             Functions::Abs      (F) =>  F.backward(grad),
//             Functions::Exp      (F) =>  F.backward(grad),
//             Functions::Log      (F) =>  F.backward(grad),
//             Functions::Neg      (F) =>  F.backward(grad),
//             Functions::Sqrt     (F) =>  F.backward(grad),
//             Functions::Square   (F) =>  F.backward(grad),
//
//             Functions::Add      (F) =>  F.backward(grad),
//             Functions::Sub      (F) =>  F.backward(grad),
//             Functions::Mul      (F) =>  F.backward(grad),
//             Functions::Div      (F) =>  F.backward(grad),
//             Functions::Pow      (F) =>  F.backward(grad),
//             Functions::Matmul   (F) =>  F.backward(grad),
//
//             Functions::Topk     (F) =>  F.backward(grad),
//             Functions::Matmax   (F) =>  F.backward(grad),
//         }
//         todo!("역전파 구현하기")
//     }
// }