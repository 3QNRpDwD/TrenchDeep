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
    use crate::tensor::{Add, Function, Square, Tensor, TensorBase, Variable};
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
        let t = add.apply(&[&x0, &x1])?;
        let y = add.apply(&[&x0, &t])?;
        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;

        println!("requires_grad 비활성화 일때 첫번째 None, None, 두번째 2.0, 1.0, 나오면 됨 아니면 버그");
        println!("{:?}, {:?}", y.grad(), t.grad());
        println!("{:?}, {:?}", x0.grad(), x1.grad());   // 버그 발생: .retain_grad() 이 True 일때 출력이 2.0, 1.0 이어야 하는데 3.0, None 이 출력됨
        // 아마 기울기 데이터가 기울기 누적 과정에서 누적되면서 3.0 이 출력되는 것 같음 다른 구조는 정상인것으로 보이는데 이 부분만 이상함

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

        println!("데이터 32 기울기 64 나오면 됨");
        println!("{:?}", y.tensor().data());
        println!("{:?}", x.grad().unwrap().data());

        Ok(())
    }

    #[test]
    fn wtf3() -> MlResult<()> { // 기울기 2, 3 나오면 됨
        let add = Add::new()?;
        let square = Square::new()?;

        let x = Arc::new(variable!(vec![vec![3.0]]));
        let y = add.apply(&[&x, &x])?; // y = add(x, x)
        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;

        println!("기울기 2, 3 나오면 됨");
        println!("{:?}", x.grad().unwrap().data());

        let y = add.apply(&[&add.apply(&[&x, &x])?, &x])?; // y = add(add(x, x), x)
        #[cfg(feature = "enable_backpropagation")]
        y.backward()?;
        println!("{:?}", x.grad().unwrap().data());

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
        println!("데이터 13, requires_grad 일때 기울기 4, 6 나오면 됨");
        println!("{:?}", z.tensor().data());

        #[cfg(feature = "requires_grad")]
        {
            println!("{:?}", x.grad().unwrap().data());
            println!("{:?}", y.grad().unwrap().data());
        }

        Ok(())
    }
}