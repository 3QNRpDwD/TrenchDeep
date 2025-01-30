use std::fmt::Display;
use std::sync::Arc;

use crate::{backend::Backend, MlResult};

mod ops;
mod broadcast;
mod creation;

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
pub struct Tensor {
    backend: Arc<dyn Backend>,
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Option<Box<Tensor>>,
    grad_fn: Option<GradFn>,
    requires_grad: bool,
}

#[derive(Clone)]
struct GradFn(Arc<dyn Fn(&Tensor) -> MlResult<()>>);

impl std::fmt::Debug for GradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradFn")
    }
}


impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl Eq for Tensor {
    // Todo: 구현 필요
}

impl PartialOrd for Tensor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait TensorBase {
    fn new(data: Vec<Vec<f32>>)                 -> MlResult<Self> where Self: Sized;
    fn from(data: Vec<f32>, shape: &[usize])    -> MlResult<Self> where Self: Sized;
    fn shape(&self)                             -> &[usize];
    fn data(&self)                              -> &[f32];
    fn get(&self, indices: &[usize])            -> Option<&f32>;
    fn index(&self, indices: &[usize])          -> Option<usize>;
    fn backend(&self)                           -> &Arc<dyn Backend>;
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

pub trait Function<T: TensorBase> {
    type Output;
    type Gradient;
    fn forward(&self) -> Self::Output;
    fn backward(&self, grad: Self::Gradient) -> Self::Output;
}

pub trait OpsLayer<T: TensorBase>{
    type Output;

    fn chk_shape(&self, other: &T) -> MlResult<()>;

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
/// A new tensor resulting from the unary operation.
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
/// A new tensor resulting from the binary operation.
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
/// A new tensor resulting from the ternary operation. For `Topk`, returns the top-k values and their indices.
/// For `Matmax`, returns the maximum values and their indices (if applicable).
#[macro_export]
macro_rules! ops {
    ($x:expr, Matmax, $y:expr, $z:expr) => {
        Matmax {
            first: $x,
            dim: $y,
            keepdim: $z,
        }.forward()
    };

    ($x:expr, Topk, $y:expr, $z:expr) => {
        Topk {
            first: $x,
            k: $y,
            sorted: $z,
        }.forward()
    };

    ($x:expr, Pow, $y:expr) => {
        Pow {
            first: $x,
            power: $y,
        }.forward()
    };

    ($x:expr, $op:ident, $y:expr) => {
        $op::new($x, $y).forward()
    };

    ($x:expr, $op:ident) => {
        $op::new($x).forward()
    };
}

/// Structure representing an exponential operation.
pub struct Exp<T: TensorBase>      { first: T }

/// Structure representing a negation operation.
pub struct Neg<T: TensorBase>      { first: T }

/// Structure representing a square root operation.
pub struct Sqrt<T: TensorBase>     { first: T }

/// Structure representing an absolute value operation.
pub struct Abs<T: TensorBase>      { first: T }

/// Structure representing a squaring operation.
pub struct Square<T: TensorBase>   { first: T }

/// Structure representing a logarithmic operation.
pub struct Log<T: TensorBase>      { first: T }

/// Structure representing an addition operation.
pub struct Add<T: TensorBase> { first: T, second: T }

/// Structure representing a subtraction operation.
pub struct Sub<T: TensorBase> { first: T, second: T }

/// Structure representing a multiplication operation.
pub struct Mul<T: TensorBase> { first: T, second: T }

/// Structure representing a division operation.
pub struct Div<T: TensorBase> { first: T, second: T }

/// Structure representing a power operation.
pub struct Pow<T: TensorBase> { first: T, power: f32 }

/// Structure representing a matrix multiplication operation.
pub struct Matmul<T: TensorBase> { first: T, second: T }

/// Structure representing a Top-k operation.
pub  struct Topk<T: TensorBase> { first: T, k: usize, sorted: bool }

/// Structure representing a matrix max operation along a dimension.
pub struct Matmax<T: TensorBase> { first: T, dim: Option<i32>, keepdim: bool }

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::{Add, TensorBase, Div, Function, Mul, OpsLayer, Sub, Tensor};

    pub fn assert_tensor_eq(tensor: Tensor, expected_tensor: Tensor, ) -> MlResult<()> {
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
