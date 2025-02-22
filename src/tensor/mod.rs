use std::{
    sync::Arc,
    fmt::{Debug, Display, Formatter, Result},
    ops::Deref
};
use crate::{backend::Backend, MlResult};

mod ops;
mod broadcast;
mod creation;

/// 다양한 텐서 연산을 위한 편리한 매크로를 제공합니다.
///
/// 이 매크로는 일반적인 텐서 연산을 수행하는 과정을 단순화하여
/// 더 편리한 문법을 제공합니다. 단항 연산과 이항 연산을 모두 지원합니다.
///
/// # supported operator
/// 이 매크로는 다음과 같은 연산들을 지원합니다:
/// - 이항 연산: `Matmul`, `Add`, `Sub`, `Mul`, `Div`
/// - 단항 연산: `Exp`, `Neg`, `Sqrt`, `Abs`, `Square`, `Log`
/// - 특수 연산: `Topk`, `Matmax`, `Pow`
///
/// # Examples
///
/// ```rust
/// use MIT::{ops, tensor::{Tensor, TensorBase}};
///
/// let mut tensor1 = Tensor::<f32>::new(vec![vec![1.0, 2.0, 3.0]]);
/// let tensor2 = Tensor::<f32>::new(vec![vec![3.0, 2.0, 1.0]]);
///
/// // 기본 산술 연산
/// let result = ops!(tensor1, Add, tensor2)?;
/// let result = ops!(tensor1, Mul, tensor2)?;
///
/// // 단항 연산
/// let result = ops!(tensor1, Sqrt)?;
/// let result = ops!(tensor1, Log)?;
///
/// // 특수 연산
/// let result = ops!(tensor1, Topk, 5, true)?; // 상위 5개 요소, 정렬됨
/// let result = ops!(tensor1, Pow, 2.0)?; // 텐서의 제곱
/// ```
///
/// # Parameters
/// - 첫 번째 매개변수는 항상 입력 텐서입니다
/// - 이항 연산의 경우, 두 번째는 연산 타입이고 세 번째는 두 번째 텐서입니다
/// - 단항 연산의 경우, 첫 텐서와 연산 타입만 필요합니다
/// - 특수 연산은 추가 매개변수가 필요할 수 있습니다 (예: Topk의 k값, Pow의 지수)
///
/// # Return
/// 지정된 연산의 순전파(forward) 결과를 반환합니다.
///
/// # Panic
/// 연산 초기화가 실패할 경우 패닉이 발생합니다 (`unwrap()`으로 감싸져 있음).
/// 이러한 에러를 직접 처리해야 하는 경우, 기본 메서드를 직접 사용하는 것을 고려하세요.
///
/// # Implementation Details
/// - 모든 연산은 원본 텐서의 형상(shape)을 유지합니다.
/// - 새로운 텐서를 생성하여 결과를 반환하므로, 원본 텐서는 변경되지 않습니다.
///
/// # Performance Considerations
/// 매 연산 시 새로운 텐서를 생성하므로
/// 대규모 텐서 연산 시 메모리 할당 오버헤드가 발생할 수 있습니다.
/// 고성능 연산이 필요한 경우 in-place 연산을 지원하는 별도 메서드 구현을 권장합니다.
#[macro_export] // 해당 매크로의 반환값에 연산자와 연산 결과를 모두 포함하도록 구조 변경을 고려중임.
macro_rules! ops {
    ($tensor:expr, Matmul, $second_tensor:expr) => {
        Matmul::new(binary!($tensor, $second_tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Topk, $k:expr, $sorted:expr) => {{
        let mut op = Topk::new(special!($tensor).unwrap()).unwrap();
        op.topk = Some(($k, $sorted));
        op.forward()
    }};

    ($tensor:expr, Matmax, $dim:expr, $keepdim:expr) => {{
        let mut op = Matmax::new(special!($tensor).unwrap()).unwrap();
        op.matmax = Some(($dim, $keepdim));
        op.forward()
    }};

    ($tensor:expr, Add, $second_tensor:expr) => {
        Add::new(binary!($tensor, $second_tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Sub, $second_tensor:expr) => {
        Sub::new(binary!($tensor, $second_tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Mul, $second_tensor:expr) => {
        Mul::new(binary!($tensor, $second_tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Div, $second_tensor:expr) => {
        Div::new(binary!($tensor, $second_tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Exp) => {
        Exp::new(unary!($tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Neg) => {
        Neg::new(unary!($tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Sqrt) => {
        Sqrt::new(unary!($tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Abs) => {
        Abs::new(unary!($tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Square) => {
        Square::new(unary!($tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Log) => {
        Log::new(unary!($tensor).unwrap()).unwrap().forward()
    };

    ($tensor:expr, Pow, $exponent:expr) => {{
        let mut op = Pow::new(unary!($tensor).unwrap()).unwrap();
        op.power = Some($exponent);
        op.forward()

    }};
}


/// 텐서와 스칼라 값 사이의 연산을 위한 매크로를 제공합니다.
///
/// # supported operator
/// ## 정방향 연산 (텐서 op 스칼라)
/// - `Add`: 텐서의 각 요소에 스칼라 값을 더함
/// - `Sub`: 텐서의 각 요소에서 스칼라 값을 뺌
/// - `Mul`: 텐서의 각 요소에 스칼라 값을 곱함
/// - `Div`: 텐서의 각 요소를 스칼라 값으로 나눔
///
/// ## 역방향 연산 (스칼라 op 텐서)
/// - `buS`: 스칼라 값에서 텐서의 각 요소를 뺌
/// - `viD`: 스칼라 값을 텐서의 각 요소로 나눔
///
/// # Examples
///
/// ```rust
/// use MIT::{scalar_ops, tensor::{Tensor, TensorBase}};
///
/// let tensor = Tensor::<f32>::new(vec![vec![1.0, 2.0, 3.0]]);
///
/// // 정방향 연산 예시
/// let result = scalar_ops!(tensor, Add, 2.0); // 모든 요소에 2.0을 더함
/// let result = scalar_ops!(tensor, Mul, 3.0); // 모든 요소에 3.0을 곱함
///
/// // 역방향 연산 예시
/// let result = scalar_ops!(5.0, buS, tensor); // 5.0에서 각 요소를 뺌
/// let result = scalar_ops!(1.0, viD, tensor); // 1.0을 각 요소로 나눔
/// ```
///
/// # Return
/// 지정된 연산의 순전파(forward) 결과를 반환합니다.
///
/// # Panic
/// 연산 초기화가 실패할 경우 패닉이 발생합니다 (`unwrap()`으로 감싸져 있음).
/// 이러한 에러를 직접 처리해야 하는 경우, 기본 메서드를 직접 사용하는 것을 고려하세요.
///
/// # Implementation Details
/// - 모든 연산은 원본 텐서의 형상(shape)을 유지합니다.
/// - 새로운 텐서를 생성하여 결과를 반환하므로, 원본 텐서는 변경되지 않습니다.
/// - Iterator와 map을 사용하여 각 요소에 대한 연산을 수행합니다.
///
/// # Performance Considerations
/// 이 매크로는 모든 연산에서 새로운 텐서를 생성합니다. 이는 텐서의 불변성을 유지하기 위한 것이지만,
/// 대규모 데이터나 빈번한 연산이 필요한 경우 성능 저하가 발생할 수 있습니다.
/// 성능이 중요한 경우, 텐서의 내부 데이터를 직접 수정하는 방식을 고려해야 할 수 있습니다.
///
/// # Optimization Considerations
/// 현재 구현은 다음과 같은 특징이 있습니다:
/// - 매 연산마다 새로운 벡터와 텐서를 할당합니다.
/// - 대규모 데이터셋에서는 메모리 사용량이 증가할 수 있습니다.
/// - 연속적인 연산의 경우 성능 저하가 누적될 수 있습니다.
///
/// 향후 개선을 위해 다음과 같은 방안을 고려할 수 있습니다:
/// - 텐서의 내부 데이터를 직접 수정하는 메서드 추가
/// - 임시 버퍼를 재사용하는 방식 도입
/// - SIMD 최적화 적용
#[macro_export]
macro_rules! scalar_ops {
    ($tensor:expr, Add, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x + $scalar).collect(), &$tensor.shape())
    };

    ($tensor:expr, Sub, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x - $scalar).collect(), &$tensor.shape())
    };

    ($tensor:expr, Mul, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x * $scalar).collect(), &$tensor.shape())
    };

    ($tensor:expr, Div, $scalar:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| x / $scalar).collect(), &$tensor.shape())
    };

    ($scalar:expr, buS, $tensor:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| $scalar - x).collect(), &$tensor.shape())
    };

    ($scalar:expr, viD, $tensor:expr) => {
        Tensor::from_vec($tensor.data().iter().map(|&x| $scalar / x).collect(), &$tensor.shape())
    };
}

#[macro_export]
macro_rules! unary {
    ($tensor:expr) => {
        UnaryOp::new($tensor.tensor)
    };
}

#[macro_export]
macro_rules! binary {
    ($first:expr, $second:expr) => {
        BinaryOp::new($first.tensor, $second.tensor)
    };
}

#[macro_export]
macro_rules! special {
    ($tensor:expr) => {
        SpecialOp::new($tensor.tensor)
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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
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

pub struct Tensor<Type: Debug>
{
    data: Vec<Type>,
    shape: Vec<usize>,
    requires_grad: bool,

    // #[cfg(feature = "enable_backpropagation")]
    // grad: Option<Box<dyn TensorBase<Type>>>,
    // #[cfg(feature = "enable_backpropagation")]
    // pub grad_fn: Option<Arc<dyn Operator<Type>>>
}

#[derive(Clone)]
pub struct ArcTensor<T>{pub tensor: Arc<dyn TensorBase<T>>}

impl ArcTensor<f32> {
    pub fn new(tensor: Tensor<f32>) -> Self {
        ArcTensor {
            tensor: Arc::new(tensor)
        }
    }
}

impl<T> Deref for ArcTensor<T> {
    type Target = dyn TensorBase<T>;

    fn deref(&self) -> &Self::Target {
        self.tensor.deref()
    }
}

impl<T: Debug + Clone> Debug for ArcTensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let tensor = self.tensor.deref();
        write!(
            f, "ArcTensor - data: {:?}, shape: {:?} requires_grad: {:?} ",
            tensor.data(), tensor.shape(), tensor.requires_grad()
        )
    }
}

impl PartialEq for Tensor<f32> {
    fn eq(&self, other: &Self) -> bool {

        self.data == other.data && self.shape == other.shape
    }
}

impl Eq for Tensor<f32> {
    // Todo: 구현 필요
}

impl PartialOrd for Tensor<f32> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl Ord for Tensor<f32> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait TensorBase<Type: Debug + Clone> {
    fn new(data: Vec<Vec<Type>>)                            -> ArcTensor<Type> where Self: Sized;
    fn from_vec(data: Vec<Type>, shape: &[usize])           -> MlResult<ArcTensor<Type>> where Self: Sized;
    // #[cfg(feature = "enable_backpropagation")]
    // fn from_grad_fn(data: Vec<Type>, shape: &[usize], grad_fn: &mut dyn Operator<f32>) -> ArcTensor<Type> where Self: Sized;

    fn shape(&self)                                         -> &[usize];
    fn data(&self)                                          -> &[Type];
    fn get(&self, indices: &[usize])                        -> Option<&Type>;
    fn index(&self, indices: &[usize])                      -> Option<usize>;
    fn chk_shape(&self, other: &dyn TensorBase<Type>)       -> MlResult<()>;
    /// Enables gradient computation for the tensor
    fn requires_grad(&self) -> bool;

    // #[cfg(feature = "enable_backpropagation")]
    // fn set_grad_fn(&mut self, grad_fn: Arc<dyn Operator<Type>>);

    #[cfg(feature = "enable_backpropagation")]
   fn grad(&self) -> Option<&dyn TensorBase<Type>>;
}

impl<Type: Debug + Clone> Debug for &dyn TensorBase<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "data: {:?}, shape: {:?} requires_grad: {:?}",
               self.data(), self.shape(), self.requires_grad()
        )
    }
}

pub trait Operator {
    fn is_start(&self) -> bool;
}

pub trait Function<T: Debug + Clone> {
    type Operator: Operator;
    type Forwarded;
    type Gradiant;

    fn new(op: Self::Operator) -> MlResult<Self> where Self: Sized;
    fn from(pr_fn: Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>, op: Self::Operator) -> MlResult<Self> where Self: Sized;
    // fn update(&mut self, first: Arc<dyn TensorBase<T>>, second: Option<Arc<dyn TensorBase<T>>>);
    fn start(op: Self::Operator)  -> MlResult<Self> where Self: Sized;
    // fn from(first: Arc<dyn Operator<T>>)  -> MlResult<Self> where Self: Sized;

    fn forward(&mut self) ->  Self::Forwarded;
    #[cfg(feature = "enable_backpropagation")]
    fn backward(&mut self, grad: &ArcTensor<T>) -> Self::Gradiant;
}

// impl<T> Debug for dyn Function<'_, T, Forwarded=(), Gradiant=()> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> Result {
//         write!(f, "Function Debug")
//     }
// }


// pub trait BroadcastLayer {
//     fn can_broadcast(&self, other: &Self) -> bool;
//     fn broadcast_shape(&self, other: &Self) -> Vec<usize>;
//     fn broadcasting<F>(self, other: Self, op: F) -> Option<Self>
//     where
//         F: Fn(f32, f32) -> f32,
//         Self: Sized;
//     fn calculate_broadcast_indices(&self, other: &Self, idx: usize, shape: &[usize]) -> Option<(usize, usize)>;
// }

#[derive(Clone)]
pub struct UnaryOp<T> { // 원래 라이프타임을 이용하여 관리했으나, 멀티스레딩 환경에서의 안전한 메모리 참조와, 사용 편의성 이슈로, Arc로 대체됨
    tensor: Arc<dyn TensorBase<T>>,
    start: bool,

    #[cfg(feature = "enable_backpropagation")]
    output: Option<Arc<dyn TensorBase<T>>>,
}

#[derive(Clone)]
pub struct BinaryOp<T> { // 원래 라이프타임을 이용하여 관리했으나, 멀티스레딩 환경에서의 안전한 메모리 참조와, 사용 편의성 이슈로, Arc로 대체됨
    first_tensor: Arc<dyn TensorBase<T>>,
    second_tensor: Arc<dyn TensorBase<T>>,
    start: bool,

    #[cfg(feature = "enable_backpropagation")]
    output: Option<Arc<dyn TensorBase<T>>>,

}

#[derive(Clone)]
pub struct SpecialOp<T> { // 원래 라이프타임을 이용하여 관리했으나, 멀티스레딩 환경에서의 안전한 메모리 참조와, 사용 편의성 이슈로, Arc로 대체됨
    tensor: Arc<dyn TensorBase<T>>,
    start: bool,

    #[cfg(feature = "enable_backpropagation")]
    output: Option<(Arc<dyn TensorBase<T>>, Arc<dyn TensorBase<T>>)>,
}

impl<T: Debug + Clone> Debug for UnaryOp<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "UnaryOp - tensor: {:?}, start: {:?}",
            self.tensor.deref(), self.start
        )
    }
}

impl<T: Debug + Clone> Debug for BinaryOp<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "BinaryOp - first_tensor: {:?} self.second_tensor: {:?}, start: {:?}",
            self.first_tensor.deref(), self.second_tensor.deref(), self.start
        )
    }
}

impl<T: Debug + Clone> Debug for SpecialOp<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "SpecialOp - tensor: {:?}, start: {:?}",
            self.tensor.deref(), self.start
        )
    }
}

/// Structure representing an exponential operation.
#[derive(Clone)]
pub struct Exp<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing a negation operation.
#[derive(Clone)]
pub struct Neg<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing a square root operation.
#[derive(Clone)]
pub struct Sqrt<T>      {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing an absolute value operation.
#[derive(Clone)]
pub struct Abs<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing a squaring operation.
#[derive(Clone)]
pub struct Square<T>    {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing a logarithmic operation.
#[derive(Clone)]
pub struct Log<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing a power operation.
#[derive(Clone)]
pub struct Pow<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: UnaryOp<T>,
    backend: Arc<dyn Backend>,
    pub power: Option<f32>,
}

/// Structure representing a Top-k operation.
#[derive(Clone)]
pub struct Topk<T>      {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: SpecialOp<T>,
    backend: Arc<dyn Backend>,
    pub topk: Option<(usize, bool)>
} // k: usize, sorted: bool

/// Structure representing a matrix max operation along a dimension.
#[derive(Clone)]
pub struct Matmax<T>    {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: SpecialOp<T>,
    backend: Arc<dyn Backend>,
    pub matmax: Option<(Option<i32>, bool)>
} // dim: (Option<i32>, keepdim: bool

/// Structure representing an addition operation.
#[derive(Clone)]
pub struct Add<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: BinaryOp<T>,
    backend: Arc<dyn Backend>,
}

/// Structure representing a subtraction operation.
#[derive(Clone)]
pub struct Sub<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: BinaryOp<T>,
    backend: Arc<dyn Backend>,}

/// Structure representing a multiplication operation.
#[derive(Clone)]
pub struct Mul<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: BinaryOp<T>,
    backend: Arc<dyn Backend>
}

/// Structure representing a division operation.
#[derive(Clone)]
pub struct Div<T>       {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: BinaryOp<T>,
    backend: Arc<dyn Backend>,
}


/// Structure representing a matrix multiplication operation.
#[derive(Clone)]
pub struct Matmul<T>    {
    pr_fn: Option<Arc<dyn Function<f32, Forwarded=MlResult<ArcTensor<f32>>, Gradiant=MlResult<(ArcTensor<f32>, ArcTensor<f32>)>, Operator=dyn Operator>>>,
    op: BinaryOp<T> ,
    backend: Arc<dyn Backend>,
}


#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::*;

    pub fn assert_tensor_eq(tensor: &ArcTensor<f32>, expected_tensor: &ArcTensor<f32>) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
        Ok(())
    }

    #[test]
    #[cfg(feature = "enable_backpropagation")]
    fn propagations() -> MlResult<()>{
        let x = Tensor::new(vec![vec![0.5]]);

        let mut A = Square::start(unary!(x)?)?;
        // 입력을 텐서가 아닌 연산자로 입력받으면 연산자가 다음(이전) 연산자를 추적할수 있으니 역전파도 가능할것 같은데
        // 시작 함수를 임의로 정의하고 시작 함수로 도달할때까지 역전파 하면 될듯 하다 물론 이게 최적인지는 잘 모르겠다
        let a = A.forward()?;       // a = A(x)

        let mut B = Exp::new(unary!(a.clone())?)?;
        let b = B.forward()?;       // b = B(a)

        let mut C = Square::new(unary!(b.clone())?)?;
        let y = C.forward()?;       //y = C(b)

        let y_grad = Tensor::new(vec![vec![1.0]]);
        let b_grad = A.backward(&y_grad)?; // Square backward
        let a_grad = B.backward(&b_grad)?; // Exp backward
        let x_grad = C.backward(&a_grad)?; // Square backward

        println!("forward\n{:?}\n{:?}\n{:?}\n", A.op, B.op, C.op);
        println!("output\n{:?}\n{:?}\n{:?}\n", a, b, y);
        println!("backward\n{:?}\n{:?}\n{:?}", b_grad, a_grad, x_grad);

        Ok(())
    }

    #[test]
    fn tensor() -> MlResult<()> {

        let t1 = Tensor::new(vec![vec![1.0, 2.0]]);
        assert_eq!(t1.data(), vec![1.0, 2.0]);
        assert_eq!(t1.shape(), vec![1, 2]);
        Ok(())
    }

    #[test]
    fn test_add() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let m_add = ops!(first.clone(), Add, second.clone())?;
        let s_add = first + second;
        let et = Tensor::<f32>::new(vec![vec![4.0, 6.0]]);

        assert_tensor_eq(&m_add, &et)?;
        assert_tensor_eq(&s_add, &et)
    }
    #[test]
    fn test_sub() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let m_sub = ops!(first.clone(), Sub, second.clone())?;
        let s_sub = first - second;
        let et = Tensor::<f32>::new(vec![vec![-2.0, -2.0]]);

        assert_tensor_eq(&m_sub, &et)?;
        assert_tensor_eq(&s_sub, &et)
    }
    #[test]
    fn test_mul() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let m_mul = ops!(first.clone(), Mul, second.clone())?;
        let s_mul = first * second;
        let et = Tensor::<f32>::new(vec![vec![3.0, 8.0]]);

        assert_tensor_eq(&m_mul, &et)?;
        assert_tensor_eq(&s_mul, &et)
    }
    #[test]
    fn test_div() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::<f32>::new(vec![vec![2.0, 4.0]]);
        let m_div = ops!(first.clone(), Div, second.clone())?;
        let s_div = first / second;
        let et = Tensor::<f32>::new(vec![vec![0.5, 0.5]]);

        assert_tensor_eq(&m_div, &et)?;
        assert_tensor_eq(&s_div, &et)
    }

    #[test]
    fn test_macro_matmul() {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0], vec![4.0]]);
        let result = ops!(first, Matmul, second).unwrap();
        assert_eq!(result.data(), vec![11.0]);
    }

    #[test]
    fn test_macro_exp() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = ops!(tensor, Exp).unwrap();
        assert_eq!(result.data(), vec![std::f32::consts::E, 7.389056]);
    }

    #[test]
    fn test_macro_neg() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = ops!(tensor, Neg).unwrap();
        assert_eq!(result.data(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_macro_sqrt() {
        let tensor = Tensor::new(vec![vec![1.0, 4.0]]);
        let result = ops!(tensor, Sqrt).unwrap();
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_macro_abs() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = ops!(tensor, Abs).unwrap();
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_macro_square() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = ops!(tensor, Square).unwrap();
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_macro_log() {
        let tensor = Tensor::new(vec![vec![1.0, std::f32::consts::E]]);
        let result = ops!(tensor, Log).unwrap();
        assert_eq!(result.data(), vec![0.0, 0.99999994]);
    }

    #[test]
    fn test_macro_pow() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = ops!(tensor, Pow, 2.0).unwrap();
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn tensor_ops_add_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::<f32>::new(vec![vec![3.0, 4.0]]);
        let result = scalar_ops!(first, Add, 2.0)?;
        // 텐서와 스칼라의 차원이 맞지 않아, 오류 발생.
        // 스칼라 연산 메서드를 따로 구현하야하나?
        assert_tensor_eq(&result, &et)
    }
    #[test]
    fn tensor_ops_sub_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::<f32>::new(vec![vec![-1.0, 0.0]]);
        let result = scalar_ops!(first, Sub, 2.0)?;

        assert_tensor_eq(&result, &et)
    }
    #[test]
    fn tensor_ops_mul_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::<f32>::new(vec![vec![2.0, 4.0]]);
        let result = scalar_ops!(first, Mul , 2.0)?;

        assert_tensor_eq(&result, &et)
    }
    #[test]
    fn tensor_ops_div_scalar() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::<f32>::new(vec![vec![0.5, 1.0]]);
        let result = scalar_ops!(first, Div , 2.0)?;

        assert_tensor_eq(&result, &et)
    }

    #[test]
    fn tensor_ops_scalar_sub() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::<f32>::new(vec![vec![1.0, 0.0]]);
        let result = scalar_ops!(2.0, buS , first)?;

        assert_tensor_eq(&result, &et)

    }
    #[test]
    fn tensor_ops_scalar_div() -> MlResult<()> {
        let first = Tensor::<f32>::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::<f32>::new(vec![vec![2.0, 1.0]]);
        let result = scalar_ops!(2.0, viD , first)?;

        assert_tensor_eq(&result, &et)
    }
}
