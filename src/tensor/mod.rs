use std::{
    sync::Arc,
    fmt::{Debug, Display, Formatter, Result},
};
use crate::{backend::Backend, MlResult};

mod ops;
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
/// use MIT::{ops, variable, tensor::{Add, Log, Mul, Sqrt, Sub}};
///
/// let tensor1 = variable!(vec![vec![1.0, 2.0, 3.0]]);
/// let tensor2 = variable!(vec![vec![3.0, 2.0, 1.0]]);
///
/// // 기본 산술 연산
/// let result = ops!(tensor1, Add, tensor2);
/// let result = ops!(tensor1, Mul, tensor2);
///
/// // 단항 연산
/// let result = ops!(tensor1, Sqrt);
/// let result = ops!(tensor1, Log);
///
/// // 특수 연산
/// let result = ops!(tensor1, Topk, 5, true); // 상위 5개 요소, 정렬됨
/// let result = ops!(tensor1, Pow, 2.0); // 텐서의 제곱
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
    ($tensor:expr, Pow, $exponent:expr) => {{
        let mut op = Pow::new().unwrap();
        op.power = Some($exponent);
        op.forward(vec![&$tensor].as_slice()).unwrap().remove(0)
    }};

    ($tensor:expr, $op:ident, $second_tensor:expr) => {
        $op::new().unwrap().forward(vec![&$tensor, &$second_tensor].as_slice()).unwrap().remove(0)
    };

    ($tensor:expr, $op:ident) => {
        $op::new().unwrap().forward(vec![&$tensor].as_slice()).unwrap().remove(0)
    };

    ($tensor:expr, Topk, $k:expr, $sorted:expr) => {{
        let mut op = Topk::new().unwrap();
        op.topk = Some(($k, $sorted));
        let mut result = op.forward(vec![&$tensor].as_slice()).unwrap();
        (result.remove(0), result.remove(0))
    }};

    ($tensor:expr, Matmax, $dim:expr, $keepdim:expr) => {{
        let mut op = Matmax::new().unwrap();
        op.matmax = Some(($dim, $keepdim));
        let mut result = op.forward(vec![&$tensor].as_slice()).unwrap();
        (result.remove(0), result.remove(0))
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
/// use MIT::{scalar_ops, variable};
///
/// let tensor = variable!(vec![vec![1.0, 2.0, 3.0]]);
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
macro_rules! variable {
    ($vec:expr) => {
        Variable::new(Tensor::new($vec))
    };

    ($data:expr, $shape:expr) => {
        Variable::new(Tensor::from_vec($data, $shape).unwrap())
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

#[derive(Debug)]
pub struct Tensor<Type: Debug> {
    data: Vec<Type>,
    shape: Vec<usize>,
}

#[derive(Debug)]
pub struct Variable<Type: Debug> {
    tensor: Tensor<Type>,
    requires_grad: bool,

    #[cfg(feature = "enable_backpropagation")]
    grad: Option<Tensor<Type>>,
    #[cfg(feature = "enable_backpropagation")]
    grad_fn: Option<Arc<dyn Function<Type>>>,
}

impl Variable<f32> {
    pub fn new(tensor: Tensor<f32>) -> Self {
        Variable {
            tensor,
            requires_grad: cfg!(feature = "enable_backpropagation"),

            #[cfg(feature = "enable_backpropagation")]
            grad: None,
            #[cfg(feature = "enable_backpropagation")]
            grad_fn: None,
        }
    }

    // pub fn from(tensor: Arc<Tensor<f32>>) -> Self {
    //     Self {
    //         tensor,
    //         requires_grad: cfg!(feature = "enable_backpropagation"),
    //
    //         #[cfg(feature = "enable_backpropagation")]
    //         grad: None,
    //         #[cfg(feature = "enable_backpropagation")]
    //         grad_fn: None,
    //     }
    // }
}

// impl<T> Deref for Variable<T> {
//     type Target = dyn TensorBase<T>;
//
//     fn deref(&self) -> &Self::Target {
//         self.tensor.deref()
//     }
// }

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
    fn new(data: Vec<Vec<Type>>)                            -> Tensor<Type> where Self: Sized;
    fn from_vec(data: Vec<Type>, shape: &[usize])           -> MlResult<Tensor<Type>> where Self: Sized;
    // #[cfg(feature = "enable_backpropagation")]
    // fn from_grad_fn(data: Vec<Type>, shape: &[usize], grad_fn: &mut dyn Operator<f32>) -> Variable<Type> where Self: Sized;

    fn shape(&self)                                         -> &[usize];
    fn data(&self)                                          -> &[Type];
    fn get(&self, indices: &[usize])                        -> Option<&Type>;
    fn index(&self, indices: &[usize])                      -> Option<usize>;
    fn chk_shape(&self, other: &dyn TensorBase<Type>)       -> MlResult<()>;
    // Enables gradient computation for the tensor
}

impl<Type: Debug + Clone> Debug for &dyn TensorBase<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f, "data: {:?}, shape: {:?}",
               self.data(), self.shape()
        )
    }
}

pub trait Function<T: Debug + Clone> {
    fn new() -> MlResult<Self> where Self: Sized;
    fn forward(&self, targets: &[&Tensor<T>])   -> MlResult<Vec<Variable<T>>>;

    #[cfg(feature = "enable_backpropagation")] // 최적화를 위해 어트리뷰트에 따라서 역전파 기능의 활성화 여부를 조절하려 했으나, 복합적인 이유(연관타입 처리)로 폐지될 에정임
    fn backward(&self, grad: &Tensor<T>)        -> MlResult<Vec<&Tensor<T>>>;

    fn backend(&self) -> &Arc<dyn Backend>;
}

#[derive(Clone)]
pub struct Exp      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Neg      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Sqrt     { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Abs      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Square   { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Log      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Pow      { backend: Arc<dyn Backend>, pub power: Option<f32> }
#[derive(Clone)]
pub struct Topk     { backend: Arc<dyn Backend>, pub topk: Option<(usize, bool)> }
#[derive(Clone)]
pub struct Matmax   { backend: Arc<dyn Backend>, pub matmax: Option<(Option<i32>, bool)> }
#[derive(Clone)]
pub struct Add      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Sub      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Mul      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Div      { backend: Arc<dyn Backend> }
#[derive(Clone)]
pub struct Matmul   { backend: Arc<dyn Backend> }

#[cfg(test)]
mod tests {
    use crate::MlResult;
    use crate::tensor::*;

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

    fn print_phase(
        phase: &str,
        x: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        y: &Tensor<f32>,
    ) {
        println!(
            "{}:\n    \
            Tensor {{ data: {:^width$?}, shape: {:^width2$?} }} ==[Square]=> Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}\n    \
            Tensor {{ data: {:^width$?}, shape: {:^width2$?} }} ==[ Exps ]=> Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}\n    \
            Tensor {{ data: {:^width$?}, shape: {:^width2$?} }} ==[Square]=> Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}\n",
            phase,
            x.data(), x.shape(),
            a.data(), a.shape(),
            a.data(), b.shape(),
            b.data(), b.shape(),
            b.data(), b.shape(),
            y.data(), y.shape(),
            width = 11,
            width2 = 3
        );
    }

    #[test]
    fn propagations_backpropagation() -> MlResult<()>{
        let x = variable!(vec![vec![0.5]]);

        let mut A = Square::new()?;
        let mut B = Exp::new()?;
        let mut C = Square::new()?;
        let a = A.forward(&[&x.tensor])?.remove(0); // a = A(x)
        let b = B.forward(&[&a.tensor])?.remove(0); // b = B(a)
        let y = C.forward(&[&b.tensor])?.remove(0); // y = C(b)

        print_phase("forward", &x.tensor, &a.tensor, &b.tensor, &y.tensor);

        #[cfg(feature = "enable_backpropagation")]
        {
            let y_grad = Tensor::new(vec![vec![1.0]]);
            let b_grad = A.backward(&y_grad)?.remove(0); // Square backward
            let a_grad = B.backward(b_grad)?.remove(0); // Exp backward
            let x_grad = C.backward(a_grad)?.remove(0); // Square backward

            print_phase("backward", &y_grad, b_grad, a_grad, x_grad);
        }

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
    fn test_add_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second =Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let s_add = first + second;

        assert_tensor_eq(&s_add.tensor, &expected)
    }

    #[test]
    fn test_sub_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![-2.0, -2.0]]);
        let s_sub = first - second;

        assert_tensor_eq(&s_sub.tensor, &expected)
    }

    #[test]
    fn test_mul_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![3.0, 8.0]]);
        let s_mul = first * second;

        assert_tensor_eq(&s_mul.tensor, &expected)
    }

    #[test]
    fn test_div_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let expected = Tensor::new(vec![vec![0.5, 0.5]]);
        let s_div = first / second;

        assert_tensor_eq(&s_div.tensor, &expected)
    }

    #[test]
    fn test_add_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let m_add = ops!(first, Add, second);

        assert_tensor_eq(&m_add.tensor, &expected)
    }

    #[test]
    fn test_sub_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![-2.0, -2.0]]);
        let m_sub = ops!(first, Sub, second);

        assert_tensor_eq(&m_sub.tensor, &expected)
    }

    #[test]
    fn test_mul_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![3.0, 8.0]]);
        let m_mul = ops!(first, Mul, second);

        assert_tensor_eq(&m_mul.tensor, &expected)
    }

    #[test]
    fn test_div_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let expected = Tensor::new(vec![vec![0.5, 0.5]]);
        let m_div = ops!(first, Div, second);

        assert_tensor_eq(&m_div.tensor, &expected)
    }

    #[test]
    fn test_matmul_macro() {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0], vec![4.0]]);
        let result = ops!(first, Matmul, second);

        assert_eq!(result.tensor.data(), vec![11.0]);
    }

    #[test]
    fn tes_macro_exp_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = ops!(tensor, Exp);
        assert_eq!(result.tensor.data(), vec![std::f32::consts::E, 7.389056]);
    }

    #[test]
    fn test_neg_macro() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = ops!(tensor, Neg);
        assert_eq!(result.tensor.data(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_sqrt_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 4.0]]);
        let result = ops!(tensor, Sqrt);
        assert_eq!(result.tensor.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_abs_macro() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = ops!(tensor, Abs);
        assert_eq!(result.tensor.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_square_macro() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = ops!(tensor, Square);
        assert_eq!(result.tensor.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_log_macro() {
        let tensor = Tensor::new(vec![vec![1.0, std::f32::consts::E]]);
        let result = ops!(tensor, Log);
        assert_eq!(result.tensor.data(), vec![0.0, 0.99999994]);
    }

    #[test]
    fn test_pow_macro() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = ops!(tensor, Pow, 2.0);
        assert_eq!(result.tensor.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn tensor_add_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::new(vec![vec![3.0, 4.0]]);
        let result = scalar_ops!(first, Add, 2.0)?;
        // 텐서와 스칼라의 차원이 맞지 않아, 오류 발생.
        // 스칼라 연산 메서드를 따로 구현하야하나?
        assert_tensor_eq(&result, &et)
    }
    #[test]
    fn tensor_sub_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::new(vec![vec![-1.0, 0.0]]);
        let result = scalar_ops!(first, Sub, 2.0)?;

        assert_tensor_eq(&result, &et)
    }
    #[test]
    fn tensor_mul_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::new(vec![vec![2.0, 4.0]]);
        let result = scalar_ops!(first, Mul , 2.0)?;

        assert_tensor_eq(&result, &et)
    }
    #[test]
    fn tensor_div_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::new(vec![vec![0.5, 1.0]]);
        let result = scalar_ops!(first, Div , 2.0)?;

        assert_tensor_eq(&result, &et)
    }

    #[test]
    fn tensor_scalar_sub() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::new(vec![vec![1.0, 0.0]]);
        let result = scalar_ops!(2.0, buS , first)?;

        assert_tensor_eq(&result, &et)

    }
    #[test]
    fn tensor_scalar_div() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let et = Tensor::new(vec![vec![2.0, 1.0]]);
        let result = scalar_ops!(2.0, viD , first)?;

        assert_tensor_eq(&result, &et)
    }
}
