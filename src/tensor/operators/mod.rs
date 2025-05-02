use super::*;
use crate::{
    backend::{
        Backend,
        CpuBackend,
        Device
    }
};
pub mod add;
pub mod sub;
pub mod mul;
pub mod div;
pub mod neg;
pub mod unary;
pub mod matmul;
pub mod topk;
pub mod matmax;
pub mod sum;
pub mod trigonometric;
pub mod reshape;
pub mod transpose;

macro_rules! define_op {
    // 기본 구조체 (매개변수 없음)
    ($name:ident) => {
        #[derive(Clone)]
        pub struct $name { backend: Arc<dyn Backend> }
    };

    // 추가 필드가 있는 구조체
    ($name:ident, $field:ident: $type:ty) => {
        #[derive(Clone)]
        pub struct $name {
            backend: Arc<dyn Backend>,
            pub $field: $type
        }
    };
}

// 기본 연산자들
define_op!(Sum);
define_op!(Exp);
define_op!(Neg);
define_op!(Sqrt);
define_op!(Abs);
define_op!(Square);
define_op!(Log);
define_op!(Add);
define_op!(Sub);
define_op!(Mul);
define_op!(Div);
define_op!(Matmul);
define_op!(Sin);  // 일반적인 사인 함수입니다.
define_op!(Cos);  // 일반적인 코사인 함수입니다.d
define_op!(Reshape);
define_op!(Transpose);
define_op!(Pow, power: Option<f32>);
define_op!(Topk, topk: Option<(usize, bool)>);
define_op!(Matmax, matmax: Option<(Option<i32>, bool)>);
define_op!(ApproxSin, threshold: f32);  // 테일러급수를 사용한 사인 함수 입니다.
define_op!(ApproxCos, threshold: f32);  // 테일러급수를 사용한 코사인 함수 입니다



/// 계산 그래프에서 연산을 정의하는 트레잇입니다.
///
/// 이 트레잇은 순전파와 역전파를 포함한 연산의 동작을 정의하며, 백엔드와의 연계를 지원합니다.
///
/// # 특징
/// 이 트레이트는 연산자의 재활용성과 확장성을 최선으로, 역전파 계산 자체를 연산자와 텐서에 종속적이지 않게 설계하는것을 목표로 합니다.
/// 따라서 연산자와 텐서 어디에도 자체적인 역전파를 지원하지 않으며, 미분값또한 직접적으로 저장하지 않습니다.
/// 대신, 연산자는 역전파를 위한 그래프를 생성하고, 그래프는 텐서의 미분값을 저장하고, 그래프를 통해 역전파를 수행합니다.
/// 이는 연산자와 텐서의 역전파 계산을 분리하여, 연산자의 재사용성과 확장성을 높이고, 역전파 계산을 효율적으로 수행할 수 있도록 합니다.
/// 단, 이러한 구조 때문에 현재 연산자 오버로딩에서의 역전파 호출 지원이 어렵습니다.
/// 따라서, 연산자 오버로딩을 통한 역전파 호출은 추후 개선을 통해 지원할 예정입니다.
///
/// # 제약
/// - `T`: `Debug + Clone` 트레잇을 구현해야 함
pub trait Function<T: Debug + Clone> {
    /// 새로운 연산 객체를 생성합니다.
    ///
    /// # 반환값
    /// - `MlResult<Self>`: 성공 시 생성된 연산 객체, 실패 시 오류
    fn new() -> MlResult<Self> where Self: Sized {
        unimplemented!("Function::new() is not implemented")
    }

    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    /// 순전파(Forward Pass)를 수행합니다.
    ///
    /// 입력 텐서들을 받아 연산을 수행하고 결과 변수를 반환합니다.
    ///
    /// # 매개변수
    /// - `targets`: 연산에 사용될 입력 텐서들의 참조 배열
    ///
    /// # 반환값
    /// - `MlResult<Vec<Variable<T>>>`: 성공 시 결과 변수 벡터, 실패 시 오류
    ///
    /// # 오류
    /// - 입력 텐서의 형태나 데이터가 연산에 적합하지 않을 경우
    fn forward(&self, _targets: &[&Tensor<T>]) -> MlResult<Vec<Tensor<T>>>{
        unimplemented!("Forward pass is not implemented")
    }

    /// 역전파(Backward Pass)를 수행합니다.
    ///
    /// 주어진 입력 텐서와 그래디언트를 기반으로 입력에 대한 그래디언트를 계산합니다.
    /// 이 메서드는 `enableBackpropagation` 기능이 활성화된 경우에만 사용 가능합니다.
    ///
    /// # 매개변수
    /// - `targets`: 역전파에 사용될 입력 텐서
    /// - `grad`: 출력에 대한 그래디언트
    ///
    /// # 반환값
    /// - `MlResult<Vec<Tensor<T>>>`: 성공 시 입력에 대한 그래디언트 벡터, 실패 시 오류
    ///
    /// # 오류
    /// - 그래디언트 계산에 실패하거나 입력이 유효하지 않을 경우
    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<T>], grad: &Tensor<T>) -> MlResult<Vec<Tensor<T>>> {
        // enableBackpropagation만 활성화된 경우의 기본 구현
        unimplemented!("Backward pass is not implemented")
    }

    /// 연산에 사용되는 백엔드를 반환합니다.
    ///
    /// # 반환값
    /// - `&Arc<dyn Backend>`: 백엔드에 대한 스마트 포인터 참조
    fn backend(&self) -> &Arc<dyn Backend> {
        unimplemented!("Function::backend() is not implemented")
    }
}

impl<Type: Debug + Clone> Debug for &dyn Function<Type> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function<{}>", std::any::type_name::<Self>())
    }
}

// Add helper method to create instances with backend
impl ApproxSin {
    pub fn with_backend(backend: Arc<dyn Backend>, threshold: f32) -> MlResult<Self> {
        Ok(Self { backend, threshold })
    }
}

impl ApproxCos {
    pub fn with_backend(backend: Arc<dyn Backend>, threshold: f32) -> MlResult<Self> {
        Ok(Self { backend, threshold })
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::tensor::operators::{Exp, Sin};
    use crate::tensor::{AutogradFunction, operators::{Add, Function, Mul, Pow, Square}, Tensor, TensorBase, Variable};
    use crate::{MlResult, variable};

    pub fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        if tensor != expected_tensor {
            return Err(format!("Expected {:?}, got {:?}", expected_tensor, tensor).into());
        }
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

    fn print_forward(
        x: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        y: &Tensor<f32>,
    ) {
        #[cfg(feature = "debugging")]
        {
            println!(
                "Forward Pass:\n    \
            Tensor {{ data: {:^width$?}, shape: {:^width2$?} }} ==[Square]=> Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}\n    \
            Tensor {{ data: {:^width$?}, shape: {:^width2$?} }} ==[ Exps ]=> Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}\n    \
            Tensor {{ data: {:^width$?}, shape: {:^width2$?} }} ==[Square]=> Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}\n",
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
    }

    fn print_backward(
        x: &Option<Tensor<f32>>,
        a: &Option<Tensor<f32>>,
        b: &Option<Tensor<f32>>,
        y: &Option<Tensor<f32>>,
    ) {
        #[cfg(feature = "debugging")]
        {
            let fmt_tensor = |t: &Option<Tensor<f32>>| {
                if let Some(tensor) = t {
                    format!(
                        "Tensor {{ data: {:^width$?}, shape: {:^width2$?} }}",
                        tensor.data(),
                        tensor.shape(),
                        width = 11,
                        width2 = 3
                    )
                } else {
                    "Tensor { data: None, shape: None }".to_string()
                }
            };

            println!(
                "Backward Pass:\n    \
        {} ==[Square]=> {}\n    \
        {} ==[ Exps ]=> {}\n    \
        {} ==[Square]=> {}\n",
                fmt_tensor(x),
                fmt_tensor(a),
                fmt_tensor(a),
                fmt_tensor(b),
                fmt_tensor(b),
                fmt_tensor(y),
            );
        }
    }

    #[test]
    fn phase_test() -> MlResult<()>{
        let square = Square::new()?;
        let exp = Exp::new()?;

        let x = variable!(vec![vec![0.5]]);
        let a = Variable::new(square.forward(&[ x.tensor() ])?.remove(0)); // a = A(x)
        let b = Variable::new(exp   .forward(&[ a.tensor() ])?.remove(0)); // b = B(a)
        let y = Variable::new(square.forward(&[ b.tensor() ])?.remove(0)); // y = C(b)

        print_forward(x.tensor(), a.tensor(), b.tensor(), y.tensor());
        assert_tensor_eq(y.tensor(), &Tensor::new(vec![vec![1.6487213]]))?;

        #[cfg(feature = "enableBackpropagation")]
        {
            y.set_grad(Tensor::new(vec![vec![1.0]]));                                  // dy = 1
            b.set_grad(square.backward(&[b.tensor()], &y.grad().unwrap())?.remove(0));   // dy/db = dy/dy * 2b
            a.set_grad(exp   .backward(&[a.tensor()], &b.grad().unwrap())?.remove(0));   // dy/da = (dy/db) * db/da
            x.set_grad(square.backward(&[x.tensor()], &a.grad().unwrap())?.remove(0));   // dy/dx = (dy/da) * da/dx

            print_backward(&y.grad(), &b.grad(), &a.grad(), &x.grad());
            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![3.2974427]]))?;
        }
        Ok(())
    }

    #[test]
    fn autograd_test() -> MlResult<()> {
        let square = Square::new()?;
        let exp = Exp::new()?;

        let x = Arc::new(variable!(vec![vec![0.5]]));
        let a = square.apply(&[&x])?;
        let b = exp   .apply(&[&a])?;
        let y = square.apply(&[&b])?;

        crate::tensor::tests::assert_tensor_eq(y.tensor(), &Tensor::new(vec![vec![1.6487213]]))?;
        print_forward(x.tensor(), a.tensor(), b.tensor(), y.tensor());

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            print_backward(&y.grad(), &b.grad(), &a.grad(), &x.grad());
            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![3.2974427]]))?;
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

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            #[cfg(feature = "requiresGrad")] {
                assert_eq!(y.grad(), Some(Tensor::new(vec![vec![1.0]])));
                assert_eq!(t.grad(), Some(Tensor::new(vec![vec![1.0]])));
            }
            #[cfg(not(feature = "requiresGrad"))] {
                assert_eq!(y.grad(), None);
                assert_eq!(t.grad(), None);
            }

            assert_tensor_eq(&x0.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
            assert_tensor_eq(&x1.grad().unwrap(), &Tensor::new(vec![vec![1.0]]))?;
        }

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
        assert_eq!(y.tensor().data(), Tensor::new(vec![vec![32.0]]).data());

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_eq!(x.grad(), Some(Tensor::new(vec![vec![64.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf3() -> MlResult<()> { // 기울기 2, 3 나오면 됨
        let add = Add::new()?;

        let x = Arc::new(variable!(vec![vec![3.0]]));
        let y = add.apply(&[&x, &x])?; // y = add(x, x)
        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;
            assert_eq!(x.grad(), Some(Tensor::new(vec![vec![2.0]])));

            let y = add.apply(&[&add.apply(&[&x, &x])?, &x])?; // y = add(add(x, x), x)
            #[cfg(feature = "enableBackpropagation")]
            y.backward()?;
            assert_eq!(x.grad(), Some(Tensor::new(vec![vec![3.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf4() -> MlResult<()> {
        let add = Add::new()?;
        let square = Square::new()?;

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let y = Arc::new(variable!(vec![vec![3.0]]));
        let z = add.apply(&[&square.apply(&[&x])?, &square.apply(&[&y])?])?; // z = add(square(x), square(y))
        assert_eq!(z.tensor().data(), Tensor::new(vec![vec![13.0]]).data());

        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_eq!(x.grad(), Some(Tensor::new(vec![vec![4.0]])));
            assert_eq!(y.grad(), Some(Tensor::new(vec![vec![6.0]])));
        }
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

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_eq!(y.tensor(), &Tensor::new(vec![vec![7.0]]));
            assert_eq!(a.grad(), Some(Tensor::new(vec![vec![2.0]])));
            assert_eq!(b.grad(), Some(Tensor::new(vec![vec![3.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf6() -> MlResult<()> {
        let mut pow = Pow::new()?;
        pow.power = Some(3.0);

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let y = pow.apply(&[&x])?; // y = x^3

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?; // dy/dx = 3x^2

            assert_eq!(y.tensor(), &Tensor::new(vec![vec![8.0]]));
            assert_eq!(x.grad(), Some(Tensor::new(vec![vec![12.0]])));
        }
        Ok(())
    }

    #[test]
    fn trigonometry_sin() -> MlResult<()> {
        let sin = Sin::new()?;

        let x = Arc::new(variable!(vec![vec![std::f32::consts::PI / 4.0]])); // 45도 (45 * 4 = 180)
        let y = sin.apply(&[&x])?;

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_tensor_eq(y.tensor(), &Tensor::new(vec![vec![std::f32::consts::FRAC_1_SQRT_2]]))?;
            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![std::f32::consts::FRAC_1_SQRT_2]]))?;
        }
        Ok(())
    }
}
