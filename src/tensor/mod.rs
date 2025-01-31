use std::fmt::Display;
use std::sync::Arc;

use crate::{backend::Backend, MlResult};

mod ops;
mod broadcast;
mod creation;

/// A macro for simplifying tensor operations in Rust.
///
/// This macro provides a convenient way to perform unary, binary, and ternary operations
/// on tensors using custom operator structures. The macro matches different patterns
/// to handle various use cases for tensor computations.
/// Handles unary operations (e.g., Neg, Abs, Sqrt).
///
/// # Arguments
///
/// * `$x`: The input tensor.
/// * `$op`: The operator struct that implements the `forward` method.
///
/// # Returns
/// A new tensor resulting from_vec the unary operation.
///
///
/// Handles binary operations (e.g., Add, Sub, Mul, Div).
///
/// # Arguments
///
/// * `$x`: The first input tensor.
/// * `$op`: The operator struct that implements the `forward` method.
/// * `$y`: The second input tensor.
///
/// # Returns
/// A new tensor resulting from_vec the binary operation.
///
///
/// Handles ternary operations for specific operators like `Topk` and `Matmax`.
///
/// # Arguments
///
/// * `$x`: The first input tensor.
/// * `$op`: The operator struct (`Topk` or `Matmax`) that implements the `forward` method.
/// * `$y`: The second parameter (e.g., `k` for `Topk`, `dim` for `Matmax`).
/// * `$z`: The third parameter (e.g., `sorted` for `Topk`, `keepdim` for `Matmax`).
///
/// # Returns
/// A new tensor resulting from_vec the ternary operation. For `Topk`, returns the top-k values and their indices.
/// For `Matmax`, returns the maximum values and their indices (if applicable).
#[macro_export]
macro_rules! ops {
    ($x:expr, $op:ident, $y:expr, $z:expr) => {
        $op::new($x, $y.unwrap(), $z.unwrap()).forward()
    };

    ($x:expr, $op:ident, $y:expr) => {
        $op::new($x, $y.unwrap()).forward()
    };

    ($x:expr, $op:ident) => {
        $op::new($x).forward()
    };
}

#[derive(Debug, Clone)]
pub enum TensorError {
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidDataLength {
        expected: usize,
        got: usize,
    },
    InvalidOperation {
        op: &'static str,
        reason: String,
    },
    InvalidAxis {
        axis: usize,
        shape: Vec<usize>,
    },
    MatrixMultiplicationError {
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },
    EmptyTensor,
}

impl std::error::Error for TensorError {}

impl Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidDataLength { expected, got } => {
                write!(f, "Invalid data length: expected {}, got {}", expected, got)
            }
            TensorError::InvalidOperation { op, reason } => {
                write!(f, "Invalid operation '{}': {}", op, reason)
            }
            TensorError::InvalidAxis { axis, shape } => {
                write!(f, "Invalid axis {} for tensor with shape {:?}", axis, shape)
            }
            TensorError::MatrixMultiplicationError {
                left_shape,
                right_shape,
            } => {
                write!(f, "Invalid dimensions for matrix multiplication: left shape {:?}, right shape {:?}", left_shape, right_shape)
            }
            TensorError::EmptyTensor => {
                write!(f, "Empty tensor")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<Type> {
    data: Type,
    shape: Vec<usize>,
    grad: Option<Box<Tensor<f32>>>,
    grad_fn: Option<GradFn<Vec<f32>>>,
    requires_grad: bool,
}

#[derive(Debug, Clone)]
pub struct Scalar<T> {
    data: T,
    shape: usize,
}

// #[derive(Debug, Clone)]
// pub struct Scalar {
//     data: f32,
//     shape: usize,
// }

#[derive(Clone)]
struct GradFn<T>(Arc<dyn Fn(&Tensor<T>) -> MlResult<()>>);

impl std::fmt::Debug for GradFn<Vec<f32>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradFn")
    }
}


impl<T> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl Eq for Tensor<Vec<f32>> {
    // Todo: 구현 필요
}

impl<T> PartialOrd for Tensor<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor<Vec<f32>> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait TensorBase<Type> {
    fn new(data: Vec<Vec<Type>>)                   -> MlResult<Self>;
    fn from_vec(data: Vec<Type>, shape: &[usize])  -> MlResult<Self>;
    fn shape(&self)                                 -> &[usize];
    fn data(&self)                                  -> &[Type];
    fn get(&self, indices: &[usize])                -> Option<&Type>;
    fn index(&self, indices: &[usize])              -> Option<usize>;
    fn chk_shape(&self, other: &Type)               -> MlResult<()>;
}

pub trait Function<F, S> where F: Sized, S: Sized {
    fn new(first: F, second: Option<S>, third: Option<bool>)
                                -> MlResult<Self>;
    fn forward(&self)           -> F;
    fn backward(&self, grad: F) -> (F, F);
    fn backend(&self)           -> &Arc<dyn Backend>;
}

trait ScalarOp<T> {
    fn apply(a: &T, b: f32) -> T;
    fn grad_self(grad: &T) -> T;
    fn grad_scalar(grad: &T) -> f32;
}

pub trait OpsLayer<T: TensorBase<T>> {
    type Output;

    // 텐서 & 스칼라 연산
    fn add_scalar(&self, scalar: f32)       -> Self::Output;
    fn sub_scalar(&self, scalar: f32)       -> Self::Output;
    fn mul_scalar(&self, scalar: f32)       -> Self::Output;
    fn div_scalar(&self, scalar: f32)       -> Self::Output;

    // 스칼라 & 텐서 연산
    fn scalar_sub(&self, scalar: f32)       -> Self::Output;
    fn scalar_div(&self, scalar: f32)       -> Self::Output;

    fn pow_scalar(&self, exponent: f32)     -> Self::Output;
    fn scalar_pow(&self, scalar: f32)       -> Self::Output;
    fn eq_scalar(&self, scalar: f32)        -> Self::Output;
}

pub trait BroadcastLayer {
    fn can_broadcast(&self, other: &Self) -> bool;
    fn broadcast_shape(&self, other: &Self) -> Vec<usize>;
    fn broadcasting<F>(self, other: Self, op: F) -> Option<Self>
    where
        F: Fn(f32, f32) -> f32,
        Self: Sized;
    fn calculate_broadcast_indices(&self, other: &Self, idx: usize, shape: &[usize]) -> Option<(usize, usize)>;
}

/// Structure representing an exponential operation.
pub struct Exp<T>     { first: T, backend: Arc<dyn Backend>}

/// Structure representing a negation operation.
pub struct Neg<T>     { first: T, backend: Arc<dyn Backend> }

/// Structure representing a square root operation.
pub struct Sqrt<T>    { first: T, backend: Arc<dyn Backend> }

/// Structure representing an absolute value operation.
pub struct Abs<T>     { first: T, backend: Arc<dyn Backend> }

/// Structure representing a squaring operation.
pub struct Square<T>  { first: T, backend: Arc<dyn Backend> }

/// Structure representing a logarithmic operation.
pub struct Log<T>     { first: T, backend: Arc<dyn Backend> }

/// Structure representing an addition operation.
pub struct Add<T>     { first: T, second: T, backend: Arc<dyn Backend> }

/// Structure representing a subtraction operation.
pub struct Sub<T>     { first: T, second: T, backend: Arc<dyn Backend> }

/// Structure representing a multiplication operation.
pub struct Mul<T>     { first: T, second: T, backend: Arc<dyn Backend> }

/// Structure representing a division operation.
pub struct Div<T>     { first: T, second: T, backend: Arc<dyn Backend> }

/// Structure representing a power operation.
pub struct Pow<T>     { first: T, second: f32, backend: Arc<dyn Backend> }

/// Structure representing a matrix multiplication operation.
pub struct Matmul<T>  { first: T, second: T, backend: Arc<dyn Backend> }

/// Structure representing a Top-k operation.
pub struct Topk<T>    { first: T, second: usize, third: bool, backend: Arc<dyn Backend> } // k: second, sorted: third

/// Structure representing a matrix max operation along a dimension.
pub struct Matmax<T>  { first: T, second: Option<i32>, third: bool, backend: Arc<dyn Backend> } // dim: second, keepdim: third

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::{Add, TensorBase, Div, Function, Mul, OpsLayer, Sub, Tensor};

    pub fn assert_tensor_eq(tensor: Tensor<f64>, expected_tensor: Tensor<f64>, ) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
        Ok(())
    }

    #[test]
    fn tensor() -> MlResult<()> {
        let t1 = Tensor::new(vec![vec![1.0, 2.0]])?;
        assert_eq!(t1.data(), vec![1.0, 2.0]);
        assert_eq!(t1.shape(), vec![1, 2]);
        Ok(())
    }

    #[test]
    fn test_add_symbol() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]])?;
        let second = Tensor::new(vec![vec![3.0, 4.0]])?;
        let et = Tensor::new(vec![vec![4.0, 6.0]])?;

        assert_tensor_eq(ops!(first, Add, second)?, et)
    }
    #[test]
    fn test_sub_symbol() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]])?;
        let second = Tensor::new(vec![vec![3.0, 4.0]])?;
        let et = Tensor::new(vec![vec![-2.0, -2.0]])?;

        assert_tensor_eq(ops!(first, Sub, second)?, et)
    }
    #[test]
    fn test_mul_symbol() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]])?;
        let second = Tensor::new(vec![vec![3.0, 4.0]])?;
        let et = Tensor::new(vec![vec![3.0, 8.0]])?;

        assert_tensor_eq(ops!(first, Mul, second)?, et)
    }
    #[test]
    fn test_div_symbol() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]])?;
        let second = Tensor::new(vec![vec![2.0, 4.0]])?;
        let et = Tensor::new(vec![vec![0.5, 0.5]])?;

        assert_tensor_eq(ops!(first, Div, second)?, et)
    }

    #[test]
    fn tensor_ops_add_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![3.0, 4.0]]).unwrap();
        assert_tensor_eq(first.add_scalar(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_sub_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![-1.0, 0.0]]).unwrap();
        assert_tensor_eq(first.sub_scalar(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_mul_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![2.0, 4.0]]).unwrap();
        assert_tensor_eq(first.mul_scalar(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_div_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![0.5, 1.0]]).unwrap();
        assert_tensor_eq(first.div_scalar(2.0).unwrap(), et)
    }

    #[test]
    fn tensor_ops_scalar_sub() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();
        assert_tensor_eq(first.scalar_sub(2.0).unwrap(), et)
    }
    #[test]
    fn tensor_ops_scalar_div() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
        let et = Tensor::new(vec![vec![2.0, 1.0]]).unwrap();
        assert_tensor_eq(first.scalar_div(2.0).unwrap(), et)
    }
}
