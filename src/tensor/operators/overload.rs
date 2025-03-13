use std::sync::Arc;
use crate::MlResult;
use crate::tensor::{Add, Div, Function, Mul, Sub, Tensor, Variable};
use crate::tensor::creation::AutogradFunction;
use crate::tensor::tests::assert_tensor_eq;

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
impl std::ops::Add for Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: Tensor<f32>) -> Self::Output {
        &self + &other
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
impl std::ops::Sub for Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: Tensor<f32>) -> Self::Output {
        &self - &other
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
impl std::ops::Mul for Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: Tensor<f32>) -> Self::Output {
        &self * &other
    }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div for Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: Tensor<f32>) -> Self::Output {
        &self / &other
    }
}

impl std::ops::Add for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn add(self, other: &Tensor<f32>) -> Self::Output {
        Add::new().unwrap().forward(&[self, other]).unwrap().remove(0).tensor
    }
}

impl std::ops::Sub for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn sub(self, other: &Tensor<f32>) -> Self::Output {
        Sub::new().unwrap().forward(&[self, other]).unwrap().remove(0).tensor
    }
}

impl std::ops::Mul for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: &Tensor<f32>) -> Self::Output {
        Mul::new().unwrap().forward(&[self, other]).unwrap().remove(0).tensor
    }
}

impl std::ops::Div for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn div(self, other: &Tensor<f32>) -> Self::Output {
        Div::new().unwrap().forward(&[self, other]).unwrap().remove(0).tensor
    }
}

/// Add trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The variable to add to self
///
/// # Returns
/// A new tensor containing the element-wise sum
///
/// # Broadcasting
/// * Supports broadcasting when adding a 1D tensor to each row of a 2D tensor
impl std::ops::Add for Variable<f32> {
    type Output = Variable<f32>;

    fn add(self: Arc<Self>, other: Arc<Variable<f32>>) -> Self::Output {
        &self + &other
    }
}

/// Subtract trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The variable to subtract from self
///
/// # Returns
/// A new tensor containing the element-wise difference
///
/// # Broadcasting
/// * Supports broadcasting when subtracting a 1D tensor from each row of a 2D tensor
impl std::ops::Sub for Variable<f32> {
    type Output = Variable<f32>;

    fn sub(self, other: Variable<f32>) -> Self::Output {
        &self - &other
    }
}

/// Multiply trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The variable to multiply with self
///
/// # Returns
/// A new tensor containing the element-wise product (Hadamard product)
///
/// # Note
/// * This performs element-wise multiplication, not matrix multiplication
/// * For matrix multiplication, use `matmul()` instead
impl std::ops::Mul for Variable<f32> {
    type Output = Variable<f32>;

    fn mul(self, other: Variable<f32>) -> Self::Output {
        &self * &other
    }
}

/// Divide trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The variable to divide self by
///
/// # Returns
/// A new tensor containing the element-wise quotient
impl std::ops::Div for Variable<f32> {
    type Output = Variable<f32>;

    fn div(self, other: Variable<f32>) -> Self::Output {
        &self / &other
    }
}

impl std::ops::Add<dyn AutogradFunction> for Arc<Variable<f32>> {
    type Output = Arc<Variable<f32>>;

    fn add(self, other: Arc<Variable<f32>>) -> Self::Output {
        Add::new().unwrap().apply(&[&self, &other]).unwrap().remove(0).tensor
    }
}

impl std::ops::Sub<dyn AutogradFunction> for Arc<Variable<f32>> {
    type Output = Arc<Variable<f32>>;

    fn sub(self, other: Arc<Variable<f32>>) -> Self::Output {
        Sub::new().unwrap().apply(&[&self, &other]).unwrap().remove(0).tensor
    }
}

impl std::ops::Mul<dyn AutogradFunction> for Arc<Variable<f32>> {
    type Output = Arc<Variable<f32>>;

    fn mul(self, other: Arc<Variable<f32>>) -> Self::Output {
        Mul::new().unwrap().apply(&[&self, &other]).unwrap().remove(0).tensor
    }
}

impl std::ops::Div<dyn AutogradFunction> for Arc<Variable<f32>> {
    type Output = Arc<Variable<f32>>;

    fn div(self, other: Arc<Variable<f32>>) -> Self::Output {
        Div::new().unwrap().apply(&[&self, &other]).unwrap().remove(0).tensor
    }
}


#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::{Tensor, TensorBase, Variable};

    pub fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
        Ok(())
    }
    #[test]
    fn test_add_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let result = first + second;

        assert_tensor_eq(&result, &expected)
    }

    #[test]
    fn test_sub_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![-2.0, -2.0]]);
        let result = first - second;

        assert_tensor_eq(&result, &expected)
    }

    #[test]
    fn test_mul_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![3.0, 8.0]]);
        let result = first * second;

        assert_tensor_eq(&result, &expected)
    }

    #[test]
    fn test_div_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let expected = Tensor::new(vec![vec![0.5, 0.5]]);
        let result = first / second;

        assert_tensor_eq(&result, &expected)
    }
}