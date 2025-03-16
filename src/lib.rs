use std::fmt::{Display, Formatter};

use crate::tensor::TensorError;

pub mod tensor;
pub mod backend;

#[derive(Debug)]
pub enum MlError {
    TensorError(TensorError),
    StringError(String),
}

impl Display for MlError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MlError::TensorError(e) => write!(f, "Tensor error: {}", e),
            MlError::StringError(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for MlError {}

impl From<TensorError> for MlError {
    fn from(error: TensorError) -> Self {
        MlError::TensorError(error)
    }
}
impl From<MlError> for TensorError {
    fn from(val: MlError) -> Self {
        match val {
            MlError::TensorError(e) => e,
            _ => unreachable!(),
        }
    }
}

impl From<String> for MlError {
    fn from(error: String) -> Self {
        MlError::StringError(error)
    }
}

impl From<&str> for MlError {
    fn from(error: &str) -> Self {
        MlError::StringError(error.to_string())
    }
}

pub type MlResult<T> = Result<T, MlError>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{MlResult, variable};
    use crate::tensor::{Add, Function, Mul, Pow, Square, Tensor, TensorBase, Variable};
    use crate::tensor::creation::AutogradFunction;

    pub fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        if tensor != expected_tensor {
            return Err(format!("Expected {:?}, got {:?}", expected_tensor, tensor).into());
        }
        Ok(())
    }

    #[test]
    fn wtf() -> MlResult<()> {
        let add = Add::new()?;

        let x0 = Arc::new(variable!(vec![vec![1.0]]));
        let x1 = Arc::new(variable!(vec![vec![1.0]]));
        let t = add.apply(&[&x0, &x1])?; // t = x0 + x1 = 2
        let y = add.apply(&[&x0, &t])?; // y = x0 + t = 3

        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;

        #[cfg(feature = "requires_grad")] {
            assert_eq!(y.grad(), Some(Tensor::new(vec![vec![1.0]])));
            assert_eq!(t.grad(), Some(Tensor::new(vec![vec![1.0]])));
        }
        #[cfg(not(feature = "requires_grad"))] {
            assert_eq!(y.grad(), None);
            assert_eq!(t.grad(), None);
        }

        assert_tensor_eq(&x0.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
        assert_tensor_eq(&x1.grad().unwrap(), &Tensor::new(vec![vec![1.0]]))?;

        // 버그 발생: .retain_grad() 이 True 일때 출력이 2.0, 1.0 이어야 하는데 3.0, None 이 출력됨
        // 아마 기울기 데이터가 기울기 누적 과정에서 누적되면서 3.0 이 출력되는 것 같음.
        // 다른 테스트는 정상인것으로 보이는데 이 부분만 이상함
        // 해결됨. 원래 최적화를 위해 동일한 텐서 입력이 들어오면 같은 노드로 처리를 했는데,
        // 이때문에 같은 값을 가진 다른 텐서를 같인 텐서에 전부 누적하는 오류가 발생했음.
        // 그런데 이 문제는 같은 내용의 다른 변수를 여러번 사용해서 발생했기 때문에,
        // 내용이 같은 변수를 한번만 사용하면 올바르게 기울기가 누적됨.
        // 로직 자체는 의도한대로 작동하는것으로 보임.
        // 만약 같은 내용의 서로 다른 변수를 사용해서 각각 변수의 기울기를 각각 다르게 누적하려면
        // 꽤나 까다로운 작업이 예상됨.
        // 이 경우 같은 내용의 변수를 서로 다르게 취급하는 옵션을 추가해야 할것으로 보임.
        // 아마도 내용자체가 아닌 변수의 메모리 아이디 등을 확인하면 될듯 함.

        Ok(())
    }

    #[test]
    fn wtf2() -> MlResult<()> { // 데이터 32 기울기 64 나오면 됨
        let add = Add::new()?;
        let square = Square::new()?;

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let a = square.apply(&[&x])?;
        let y = add.apply(&[&square.apply(&[&a])?, &square.apply(&[&a])?])?;

        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;

        assert_eq!(y.tensor().data(), Tensor::new(vec![vec![32.0]]).data());
        assert_eq!(x.grad(), Some(Tensor::new(vec![vec![64.0]])));
        Ok(())
    }

    #[test]
    fn wtf3() -> MlResult<()> { // 기울기 2, 3 나오면 됨
        let add = Add::new()?;

        let x = Arc::new(variable!(vec![vec![3.0]]));
        let y = add.apply(&[&x, &x])?; // y = add(x, x)
        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;
        assert_eq!(x.grad(), Some(Tensor::new(vec![vec![2.0]])));

        let y = add.apply(&[&add.apply(&[&x, &x])?, &x])?; // y = add(add(x, x), x)
        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;
        assert_eq!(x.grad(), Some(Tensor::new(vec![vec![3.0]])));
        Ok(())
    }

    #[test]
    fn wtf4() -> MlResult<()> {
        let add = Add::new()?;
        let square = Square::new()?;

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let y = Arc::new(variable!(vec![vec![3.0]]));
        let z = add.apply(&[&square.apply(&[&x])?, &square.apply(&[&y])?])?; // z = add(square(x), square(y))

        #[cfg(feature = "enable_backpropagation")]
        z.backward()?;

        assert_eq!(z.tensor().data(), Tensor::new(vec![vec![13.0]]).data());
        assert_eq!(x.grad(), Some(Tensor::new(vec![vec![4.0]])));
        assert_eq!(y.grad(), Some(Tensor::new(vec![vec![6.0]])));
        Ok(())
    }

    #[test]
    fn wtf5() -> MlResult<()> {
        let add = Add::new()?;
        let mul = Mul::new()?;

        let a = Arc::new(variable!(vec![vec![3.0]]));
        let b = Arc::new(variable!(vec![vec![2.0]]));
        let c = Arc::new(variable!(vec![vec![1.0]]));

        let y = add.apply(&[&mul.apply(&[&a, &b])?, &c])?;

        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;

        assert_eq!(y.tensor(), &Tensor::new(vec![vec![7.0]]));
        assert_eq!(a.grad(), Some(Tensor::new(vec![vec![2.0]])));
        assert_eq!(b.grad(), Some(Tensor::new(vec![vec![3.0]])));
        Ok(())
    }

    #[test]
    fn wtf6() -> MlResult<()> {
        let mut pow = Pow::new()?;
        pow.power = Some(3.0);

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let y = pow.apply(&[&x])?; // y = x^3

        #[cfg(feature = "enable_backpropagation")]
        y.backward()?; // dy/dx = 3x^2

        assert_eq!(y.tensor(), &Tensor::new(vec![vec![8.0]]));
        assert_eq!(x.grad(), Some(Tensor::new(vec![vec![12.0]])));
        Ok(())
    }
}