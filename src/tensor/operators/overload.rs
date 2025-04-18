use super::*;

/// Add trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to add to self
///
/// # Returns
/// A new tensor containing the element-wise sum
///
/// # Broadcasting
/// * Supports broadcasting when adding a 1D tensor to each row of a 2D tensor
impl std::ops::Add<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: &Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: &Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}

/// Subtract trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to subtract from self
///
/// # Returns
/// A new tensor containing the element-wise difference
///
/// # Broadcasting
/// * Supports broadcasting when subtracting a 1D tensor from each row of a 2D tensor
impl std::ops::Sub<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: &Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: &Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

/// Multiply trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to multiply with self
///
/// # Returns
/// A new tensor containing the element-wise product (Hadamard product)
///
/// # Note
/// * This performs element-wise multiplication, not matrix multiplication
/// * For matrix multiplication, use `matmul()` instead
impl std::ops::Mul<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: Tensor<f32>) -> Self::Output {
        Mul::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: &Tensor<f32>) -> Self::Output {
        Mul::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: &Tensor<f32>) -> Self::Output {
        Mul::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: Tensor<f32>) -> Self::Output {
        Mul::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: &Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: &Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Div<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Neg for Tensor<f32> {
    type Output = Tensor<f32>;

    fn neg(self) -> Self::Output {
        Neg::new().unwrap().forward(&[&self]).unwrap().remove(0)
    }
}

impl std::ops::Neg for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn neg(self) -> Self::Output {
        Neg::new().unwrap().forward(&[self]).unwrap().remove(0)
    }
}

// impl std::tensor_ops::Add for &ArcVariable<f32> {
//     type Output = Arc<Variable<f32>>;
//
//     fn add(self, other: &ArcVariable<f32>) -> Self::Output {
//         Add::new().unwrap().apply(&[self.inner(), other.inner()]).unwrap()
//     }
// }

#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, TensorBase, Variable};
    use crate::MlResult;

    pub fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
        Ok(())
    }

    pub fn assert_variable_eq(variable: &Variable<f32>, expected_variable: &Variable<f32>) -> MlResult<()> {
        assert_eq!(variable.tensor.data(), expected_variable.tensor.data());
        assert_eq!(variable.tensor.shape(), expected_variable.tensor.shape());
        Ok(())
    }

    #[test]
    fn tensor_add_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let result = first + second;

        assert_tensor_eq(&result, &expected)
    }

    #[test]
    fn tensor_sub_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let result = first - second;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![-2.0, -2.0]]))
    }

    #[test]
    fn tensor_mul_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let result = first * second;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![3.0, 8.0]]))
    }

    #[test]
    fn tensor_div_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let result = first / second;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![0.5, 0.5]]))
    }

    #[test]
    fn tensor_neg_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);

        assert_tensor_eq(&-first, &Tensor::new(vec![vec![-1.0, -2.0]]))
    }
}