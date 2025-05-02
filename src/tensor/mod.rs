use std::fmt::Display;
use std::{
    fmt::{Debug, Formatter, Result},
    sync::{Arc}
};

pub mod creation;
pub mod operators;
pub mod display;
pub mod graph;

use crate::{MlError, MlResult, tensor::operators::Function, TensorError};

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
#[macro_export] // 해당 매크로의 존재 의의가 다소 부족함. 제거또는 구조 변경을 고려.
macro_rules! tensor_ops {
    ($tensor:expr, Pow, $exponent:expr) => {{
        let mut op = Pow::new().unwrap();
        op.power = Some($exponent);
        op.forward(&[&$tensor]).unwrap().remove(0)
    }};

    ($tensor:expr, $op:ident, $second_tensor:expr) => {
        $op::new().unwrap().forward(&[&$tensor, &$second_tensor]).unwrap().remove(0)
    };

    ($tensor:expr, $op:ident) => {
        $op::new().unwrap().forward(&[&$tensor]).unwrap().remove(0)
    };

    ($tensor:expr, Topk, $k:expr, $sorted:expr) => {{
        let mut op = Topk::new().unwrap();
        op.topk = Some(($k, $sorted));
        let mut result = op.forward(&[&$tensor]).unwrap();
        (result.remove(0), result.remove(0))
    }};

    ($tensor:expr, Matmax, $dim:expr, $keepdim:expr) => {{
        let mut op = Matmax::new().unwrap();
        op.matmax = Some(($dim, $keepdim));
        let mut result = op.forward(&[&$tensor]).unwrap();
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

/// 다차원 배열을 나타내는 텐서 구조체입니다.
///
/// 이 구조체는 데이터와 그 형태(shape)를 저장하여 수학적 연산을 수행하는 데 사용됩니다.
/// 제네릭 타입 `Type`을 통해 다양한 데이터 타입을 지원합니다.
///
/// # 필드
/// - `data`: 텐서의 데이터를 1차원 벡터 형태로 저장
/// - `shape`: 텐서의 차원을 나타내는 크기 배열 (예: `[행, 열]` 또는 `[채널, 높이, 너비]`)
#[derive(Debug, Clone)]
pub struct Tensor<Type> {
    data: Vec<Type>,
    shape: Vec<usize>,
}

/// 계산 그래프에서 사용되는 변수 구조체입니다.
///
/// 이 구조체는 텐서와 그래디언트 계산 여부를 관리하며, 역전파를 지원하는 경우 그래디언트를 저장합니다.
///
/// # 필드
/// - `tensor`: 변수의 값이 담긴 텐서
/// - `requires_grad`: 그래디언트 계산이 필요한지 여부
/// - `grad`: 역전파를 위한 그래디언트 (옵션으로 저장되며, `RefCell`로 래핑되어 가변성 제공)
///   - `enableBackpropagation` 기능이 활성화된 경우에만 포함됨
pub struct Variable<Type> {
    tensor: Tensor<Type>,
    requires_grad: bool,

    #[cfg(all(feature = "enableBackpropagation"))]
    grad: std::cell::RefCell<Option<Tensor<Type>>>,
}

/// 계산 그래프에서 노드의 고유 식별자를 나타내는 타입 별칭입니다.
///
/// 이 타입은 `usize`를 기반으로 하며, 역전파 기능이 활성화된 경우에만 정의됩니다.
///
/// # 사용처
/// - `ComputationNode`와 `ComputationGraph`에서 노드를 식별하는 데 사용
#[cfg(feature = "enableBackpropagation")]
type NodeId<T> = *const Variable<T>;
#[cfg(feature = "enableBackpropagation")]
type FuncId<T> = *const dyn Function<T>;

/// 계산 그래프의 개별 노드를 나타내는 구조체입니다.
///
/// 이 구조체는 변수, 연산 함수, 입력 노드 정보를 포함하며, 역전파를 위한 계산 단위를 정의합니다.
/// 제네릭 타입 `T`는 디버깅과 복제를 지원해야 합니다.
///
/// # 필드
/// - `id`: 노드의 고유 식별자
/// - `variable`: 노드가 나타내는 변수 (스마트 포인터로 감싸짐)
/// - `function`: 노드에서 수행되는 연산 함수 (옵션, 동적 디스패치 지원)
/// - `inputs`: 이 노드의 입력으로 사용되는 다른 노드들의 ID 목록
///
#[cfg(feature = "enableBackpropagation")]
pub(crate) struct ComputationNode<T: Debug + Clone> {
    id: NodeId<T>,
    variable: Arc<Variable<T>>,
    function: Option<Arc<dyn Function<T>>>,
    inputs: Vec<NodeId<T>>,
    is_life: bool,
}

/// 계산 그래프 전체를 관리하는 구조체입니다.
///
/// 이 구조체는 노드 집합과 위상 정렬 정보를 저장하며, 역전파를 수행하는 데 필요한 데이터를 유지합니다.
/// 제네릭 타입 `T`는 디버깅과 복제를 지원해야 합니다.
///
/// # 필드
/// - `nodes`: 노드 ID와 `ComputationNode`를 매핑하는 해시맵
/// - `next_id`: 다음에 생성될 노드에 부여할 ID
/// - `topo_sorted`: 위상 정렬된 노드 ID 목록
/// - `sorted`: 위상 정렬이 완료되었는지 여부
#[cfg(feature = "enableBackpropagation")]
pub(crate) struct ComputationGraph<T: Debug + Clone> {
    nodes: std::collections::HashMap<NodeId<T>, ComputationNode<T>>,
    topo_sorted: Vec<NodeId<T>>,
    sorted: bool,
}

impl PartialEq for Tensor<f32> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

#[cfg(feature = "enableBackpropagation")]
impl PartialEq for &Variable<f32> {
    fn eq(&self, other: &&Variable<f32>) -> bool {
        self.tensor == other.tensor &&
            self.requires_grad == other.requires_grad &&
            self.grad == other.grad
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
/// 텐서의 기본 동작을 정의하는 트레잇입니다.
///
/// 이 트레잇은 텐서 생성, 데이터 접근, 형태 확인 등의 기본 기능을 제공합니다.
/// 제네릭 타입 `Type`은 디버깅과 복제를 지원해야 합니다.
///
/// # 제약
/// - `Type`: `Debug + Clone` 트레잇을 구현해야 함
pub trait TensorBase<Type: Debug + Clone> {
    /// 2차원 벡터 데이터를 기반으로 새로운 텐서를 생성합니다.
    ///
    /// # 매개변수
    /// - `data`: 텐서의 데이터를 나타내는 2차원 벡터
    ///
    /// # 반환값
    /// - `Tensor<Type>`: 생성된 텐서 객체
    fn new(_data: Vec<Vec<Type>>) -> Self where Self: Sized {
        unimplemented!(" TensorBase::new() is not implemented ")
    }

    /// 1차원 벡터와 형태를 기반으로 새로운 텐서를 생성합니다.
    ///
    /// # 매개변수
    /// - `data`: 텐서의 데이터를 나타내는 1차원 벡터
    /// - `shape`: 텐서의 차원을 나타내는 크기 배열
    ///
    /// # 반환값
    /// - `MlResult<Tensor<Type>>`: 성공 시 생성된 텐서, 실패 시 오류
    ///
    /// # 오류
    /// - 데이터 길이와 형태가 일치하지 않을 경우
    fn from_vec(_data: Vec<Type>, _shape: &[usize]) -> MlResult<Self> where Self: Sized {
        unimplemented!(" TensorBase::from_vec() is not implemented ")
    }

    /// 텐서의 형태를 반환합니다.
    ///
    /// # 반환값
    /// - `&[usize]`: 텐서의 차원을 나타내는 슬라이스
    fn shape(&self) -> &[usize] {
        unimplemented!(" TensorBase::shape() is not implemented ")
    }

    /// 텐서의 데이터를 반환합니다.
    ///
    /// # 반환값
    /// - `&[Type]`: 텐서의 데이터를 나타내는 슬라이스
    fn data(&self) -> &[Type] {
        unimplemented!(" TensorBase::data() is not implemented ")
    }

    /// 주어진 인덱스에서 텐서의 값을 반환합니다.
    ///
    /// # 매개변수
    /// - `indices`: 텐서 내 특정 위치를 가리키는 인덱스 배열
    ///
    /// # 반환값
    /// - `Option<&Type>`: 해당 위치의 값에 대한 참조, 유효하지 않은 인덱스면 `None`
    fn get(&self, _indices: &[usize]) -> Option<&Type> {
        unimplemented!(" TensorBase::get() is not implemented ")
    }

    /// 주어진 인덱스를 데이터 벡터 내의 오프셋으로 변환합니다.
    ///
    /// # 매개변수
    /// - `indices`: 텐서 내 특정 위치를 가리키는 인덱스 배열
    ///
    /// # 반환값
    /// - `Option<usize>`: 데이터 벡터 내 해당 위치의 오프셋, 유효하지 않은 인덱스면 `None`
    fn index(&self, _indices: &[usize]) -> Option<usize> {
        unimplemented!(" TensorBase::index() is not implemented ")
    }

    /// 두 텐서의 형태가 동일한지 확인합니다.
    ///
    /// # 매개변수
    /// - `other`: 비교 대상 텐서 (동적 디스패치 지원)
    ///
    /// # 반환값
    /// - `MlResult<()>`: 형태가 일치하면 `Ok(())`, 그렇지 않으면 오류
    ///
    /// # 오류
    /// - 두 텐서의 형태가 일치하지 않을 경우
    fn chk_shape(&self, other: &dyn TensorBase<Type>) -> MlResult<()> {
        if self.shape() == other.shape() {
            Ok(())
        } else {
            Err(MlError::TensorError(TensorError::InvalidShape {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            }))
        }
    }
}

/// 자동 미분(autograd)을 지원하는 함수 트레잇
///
/// 이 트레잇은 Function<f32>와 Clone을 구현하는 타입에 자동 미분 기능을 추가합니다.
/// 신경망의 순전파(forward pass)와 역전파(backward pass)를 연결하는 함수를 생성하는 역할을 수행합니다.
///
/// # 주요 기능
/// - 입력 변수들로부터 계산 결과 생성
/// - enableBackpropagation 기능 활성화 시 자동으로 역전파 그래프 구성
/// - 연산 결과에 그라데이션 함수 연결
///
/// # 메서드
///
/// ## apply(&self, inputs: &[&Arc<Variable<f32>>]) -> MlResult<Arc<Variable<f32>>>
/// 입력 변수 슬라이스를 받아 다음 단계를 처리합니다:
/// 1. 모든 입력 변수에서 텐서 추출
/// 2. forward 메소드를 호출하여 순전파 수행
/// 3. (조건부) 역전파를 위한 계산 그래프 구성
///
/// ### 기능 플래그 동작
/// - #[cfg(feature = "enableBackpropagation")] 활성화 시:
/// - 단일 입력/출력 연산만 지원 (입력 슬라이스 길이 == 1)
/// - 계산 결과에 그라데이션 함수와 입력 변수들을 연결
/// - 기능 비활성화 시 기본 텐서 연산만 수행
///
/// # 구현 참고사항
/// - Function<f32> + Clone + 'static을 구현하는 모든 타입에 자동 구현 제공
/// - 사용자 정의 연산 구현 시 forward 메소드의 정확한 구현이 필요
///
/// # 제약 사항
/// - 현재 버전에서는 다중 입력/출력에 대한 역전파를 지원하지 않음
/// - f32 데이터 타입 전용으로 특화됨
pub trait AutogradFunction<Type: Debug + Clone>: Function<Type> + Clone where Self: 'static {
    fn apply(&self, _inputs: &[&Arc<Variable<Type>>]) -> MlResult<Arc<Variable<Type>>> {
        unimplemented!(" AutogradFunction::apply() not implemented for this type")
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::operators::{Abs, Add, Div, Exp, Function, Log, Matmul, Mul, Neg, Pow, Sqrt, Square, Sub};
    use crate::tensor::{Tensor, TensorBase};
    use crate::MlResult;

    pub fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        assert_eq!(tensor.data(), expected_tensor.data());
        assert_eq!(tensor.shape(), expected_tensor.shape());
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
    fn test_add_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let m_add = tensor_ops!(first, Add, second);

        assert_tensor_eq(&m_add, &expected)
    }

    #[test]
    fn test_sub_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![-2.0, -2.0]]);
        let m_sub = tensor_ops!(first, Sub, second);

        assert_tensor_eq(&m_sub, &expected)
    }

    #[test]
    fn test_mul_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![3.0, 8.0]]);
        let m_mul = tensor_ops!(first, Mul, second);

        assert_tensor_eq(&m_mul, &expected)
    }

    #[test]
    fn test_div_macro() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let m_div = tensor_ops!(first, Div, second);

        assert_tensor_eq(&m_div, &Tensor::new(vec![vec![0.5, 0.5]]))
    }

    #[test]
    fn test_matmul_macro() {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0], vec![4.0]]);
        let result = tensor_ops!(first, Matmul, second);

        assert_eq!(result.data(), vec![11.0]);
    }

    #[test]
    fn tes_macro_exp_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = tensor_ops!(tensor, Exp);
        assert_eq!(result.data(), vec![std::f32::consts::E, 7.389056]);
    }

    #[test]
    fn test_neg_macro() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = tensor_ops!(tensor, Neg);
        assert_eq!(result.data(), vec![-1.0, 2.0]);
    }

    #[test]
    fn test_sqrt_macro() {
        let tensor = Tensor::new(vec![vec![1.0, 4.0]]);
        let result = tensor_ops!(tensor, Sqrt);
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_abs_macro() {
        let tensor = Tensor::new(vec![vec![1.0, -2.0]]);
        let result = tensor_ops!(tensor, Abs);
        assert_eq!(result.data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_square_macro() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = tensor_ops!(tensor, Square);
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_log_macro() {
        let tensor = Tensor::new(vec![vec![1.0, std::f32::consts::E]]);
        let result = tensor_ops!(tensor, Log);
        assert_eq!(result.data(), vec![0.0, 0.99999994]);
    }

    #[test]
    fn test_pow_macro() {
        let tensor = Tensor::new(vec![vec![2.0, 3.0]]);
        let result = tensor_ops!(tensor, Pow, 2.0);
        assert_eq!(result.data(), vec![4.0, 9.0]);
    }

    #[test]
    fn tensor_add_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Add, 2.0)?;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![3.0, 4.0]]))
    }
    #[test]
    fn tensor_sub_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Sub, 2.0)?;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![-1.0, 0.0]]))
    }
    #[test]
    fn tensor_mul_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Mul , 2.0)?;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![2.0, 4.0]]))
    }
    #[test]
    fn tensor_div_scalar() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(first, Div , 2.0)?;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![0.5, 1.0]]))
    }

    #[test]
    fn tensor_scalar_sub() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(2.0, buS , first)?;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![1.0, 0.0]]))

    }
    #[test]
    fn tensor_scalar_div() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let result = scalar_ops!(2.0, viD , first)?;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![2.0, 1.0]]))
    }
}
