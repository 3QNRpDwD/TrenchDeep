use super::*;

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
        pub struct $name {
            backend: Arc<dyn Backend>,
            node_id: NodeId,
        }
    };

    // 추가 필드가 있는 구조체
    ($name:ident, $field:ident: $type:ty) => {
        #[derive(Clone)]
        pub struct $name {
            backend: Arc<dyn Backend>,
            node_id: NodeId,
            pub $field: $type
        }
    };
}

#[macro_export]
macro_rules! register_operator {
    ($name:ident) => {
        {
        use crate::tensor::NODE_ID_GEN;
        use crate::tensor::OPERATOR_STORAGE;
        use crate::backend::CpuBackend;
        use crate::backend::Device;
            {
                OPERATOR_STORAGE.with(|ops| {
                    let my = stringify!($name);
                    let mut ops = ops.borrow_mut();
                    match ops.contains_key(my) {
                        true => Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id())),
                        false => {
                            ops.insert(
                                String::from(my),
                                Box::new($name {
                                    backend: Arc::new(CpuBackend::new()?),
                                    node_id: NODE_ID_GEN.next(),
                                })
                            );
                            Ok(GlobalFunction::new(String::from(my), *ops.get(my).unwrap().node_id()))
                        }
                    }
                })
            }
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
define_op!(Transpose, dims: (i32, i32));
define_op!(Pow, power: Option<f32>);
define_op!(Topk, topk: Option<(usize, bool)>);
define_op!(Matmax, matmax: Option<(Option<i32>, bool)>);
define_op!(ApproxSin, threshold: f32);  // 테일러급수를 사용한 사인 함수 입니다.
define_op!(ApproxCos, threshold: f32);  // 테일러급수를 사용한 코사인 함수 입니다

pub trait Function {
    fn new() -> MlResult<GlobalFunction> where Self: Sized {
        unimplemented!("{} Function::new() is not implemented", std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown"))
    }

    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    fn forward(&mut self, _targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>>{
        unimplemented!("{} Forward pass is not implemented", self.type_name())
    }

    fn assign_forward(&mut self, _targets: &[&dyn TensorBase], tensor_id: NodeId) -> MlResult<Vec<Tensor>> {
        unimplemented!("{} Forward pass is not implemented", self.type_name())
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&mut self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        // enableBackpropagation만 활성화된 경우의 기본 구현
        unimplemented!("{} Backward pass is not implemented", self.type_name())
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        unimplemented!("{} Function::backend() is not implemented", self.type_name())
    }
    
    fn node_id(&self) -> &NodeId {
        unimplemented!("{} Function::node_id() is not implemented", self.type_name())
    }
}



impl Debug for &dyn Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Function<{}>", std::any::type_name::<Self>())
    }
}

// Add helper method to create instances with backend
impl ApproxSin {
    pub fn with_backend(backend: Arc<dyn Backend>, threshold: f32) -> MlResult<Self> {
        Ok(Self {
            backend,
            threshold,
            node_id: NODE_ID_GEN.next(),
        })
    }
}

impl ApproxCos {
    pub fn with_backend(backend: Arc<dyn Backend>, threshold: f32) -> MlResult<Self> {
        Ok(Self {
            backend,
            threshold,
            node_id: NODE_ID_GEN.next(),
        })
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::tensor::operators::{Div, Exp, Neg, Sin, Sub};
    use crate::tensor::{AutogradFunction, operators::{Add, Function, Mul, Pow, Square}, Tensor, TensorBase, Variable};
    use crate::{scalar, variable, MlResult};

    pub fn assert_tensor_eq(tensor: &dyn TensorBase, expected_tensor: &dyn TensorBase) -> MlResult<()> {
        if tensor.data() != expected_tensor.data() && tensor.shape() != expected_tensor.shape() {
            return Err(format!("Expected {:?}, got {:?}", expected_tensor, tensor).into());
        }
        Ok(())
    }

    pub fn assert_variable_eq(variable: &Variable, expected_variable: &Variable) -> MlResult<()> {
        assert_eq!(variable.tensor.data(), expected_variable.tensor.data());
        assert_eq!(variable.tensor.shape(), expected_variable.tensor.shape());
        Ok(())
    }

    #[test]
    fn tensor_add_operator() -> MlResult<()> {
        Add::new()?;
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let result = first + second;

        assert_tensor_eq(&result, &expected)
    }

    #[test]
    fn tensor_sub_operator() -> MlResult<()> {
        Sub::new()?;
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let result = first - second;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![-2.0, -2.0]]))
    }

    #[test]
    fn tensor_mul_operator() -> MlResult<()> {
        Mul::new()?;
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let result = first * second;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![3.0, 8.0]]))
    }

    #[test]
    fn tensor_div_operator() -> MlResult<()> {
        Div::new()?;
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![2.0, 4.0]]);
        let result = first / second;

        assert_tensor_eq(&result, &Tensor::new(vec![vec![0.5, 0.5]]))
    }

    #[test]
    fn tensor_neg_operator() -> MlResult<()> {
        Neg::new()?;
        let first = Tensor::new(vec![vec![1.0, 2.0]]);

        assert_tensor_eq(&-first, &Tensor::new(vec![vec![-1.0, -2.0]]))
    }

    fn print_forward(
        x: &dyn TensorBase,
        a: &dyn TensorBase,
        b: &dyn TensorBase,
        y: &dyn TensorBase,
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
        x: Option<&dyn TensorBase>,
        a: Option<&dyn TensorBase>,
        b: Option<&dyn TensorBase>,
        y: Option<&dyn TensorBase>,
    ) {
        #[cfg(feature = "debugging")]
        {
            let fmt_tensor = |t: Option<&dyn TensorBase>| {
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
        let mut square = Square::new()?;
        let mut exp = Exp::new()?;

        let x = scalar!(0.5);
        let a = square.forward(&[ &x ])?.remove(0); // a = A(x)
        let b = exp   .forward(&[ &a ])?.remove(0); // b = B(a)
        let y = square.forward(&[ &b ])?.remove(0); // y = C(b)

        print_forward(&x, &a, &b, &y);
        assert_tensor_eq(&y, &Tensor::new(vec![vec![1.6487213]]))?;

        #[cfg(feature = "enableBackpropagation")]
        {
            let dy = scalar!(1.0);                              // dy = 1
            let db = square.backward(&[&b], &dy)?.remove(0);   // dy/db = dy/dy * 2b
            let da = exp   .backward(&[&a], &db)?.remove(0);   // dy/da = (dy/db) * db/da
            let dx = square.backward(&[&x], &da)?.remove(0);   // dy/dx = (dy/da) * da/dx

            print_backward(Some(&dy), Some(&db), Some(&da), Some(&dx));
            assert_tensor_eq(&dx, &Tensor::new(vec![vec![3.2974427]]))?;
        }
        Ok(())
    }

    #[test]
    fn autograd_test() -> MlResult<()> {
        let mut square = Square::new()?;
        let mut exp = Exp::new()?;

        let x = Arc::new(variable!(vec![vec![0.5]]));
        let a = square.apply(&[&x])?;
        let b = exp   .apply(&[&a])?;
        let y = square.apply(&[&b])?;

        crate::tensor::tests::assert_tensor_eq(y.tensor(), &Tensor::new(vec![vec![1.6487213]]))?;
        print_forward(x.tensor(), a.tensor(), b.tensor(), y.tensor());


        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;
            let dy = y.grad().unwrap();                              // dy = 1
            let db = b.grad().unwrap();   // dy/db = dy/dy * 2b
            let da = a.grad().unwrap();   // dy/da = (dy/db) * db/da
            let dx = x.grad().unwrap();   // dy/dx = (dy/da) * da/dx

            print_backward(Some(dy), Some(db), Some(da), Some(dx));
            assert_tensor_eq(x.grad().unwrap(), &Tensor::new(vec![vec![3.2974427]]))?;
        }
        Ok(())
    }

    #[test]
    fn wtf() -> MlResult<()> {
        let mut add = Add::new()?;

        let x0 = Arc::new(variable!(vec![vec![1.0]]));
        let x1 = Arc::new(variable!(vec![vec![1.0]]));
        let t = add.apply(&[&x0, &x1])?; // t = x0 + x1 = 2
        let y = add.apply(&[&x0, &t])?; // y = x0 + t = 3

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            #[cfg(feature = "requiresGrad")] {
                assert_eq!(y.grad(), Some(&Tensor::new(vec![vec![1.0]])));
                assert_eq!(t.grad(), Some(&Tensor::new(vec![vec![1.0]])));
            }
            #[cfg(not(feature = "requiresGrad"))] {
                assert_eq!(y.grad(), None);
                assert_eq!(t.grad(), None);
            }

            assert_tensor_eq(x0.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
            assert_tensor_eq(x1.grad().unwrap(), &Tensor::new(vec![vec![1.0]]))?;
        }

        // 버그 발생: .is_retain_grad() 이 True 일때 출력이 2.0, 1.0 이어야 하는데 3.0, None 이 출력됨
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
        let mut add = Add::new()?;
        let mut square = Square::new()?;

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let a = square.apply(&[&x])?;
        let y = add.apply(&[&square.apply(&[&a])?, &square.apply(&[&a])?])?;
        assert_eq!(y.tensor().data(), Tensor::new(vec![vec![32.0]]).data());

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_eq!(x.grad(), Some(&Tensor::new(vec![vec![64.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf3() -> MlResult<()> { // 기울기 2, 3 나오면 됨
        let mut add = Add::new()?;

        let x = Arc::new(variable!(vec![vec![3.0]]));
        let y = add.apply(&[&x, &x])?; // y = add(x, x)
        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;
            assert_eq!(x.grad(), Some(&Tensor::new(vec![vec![2.0]])));

            x.clear_grad();
            y.clear_grad();

            let t = add.apply(&[&x, &x])?;
            let y = add.apply(&[&t, &x])?; // y = add(add(x, x), x)
            #[cfg(feature = "enableBackpropagation")]
            y.backward()?;
            assert_eq!(x.grad(), Some(&Tensor::new(vec![vec![3.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf4() -> MlResult<()> {
        let mut add = Add::new()?;
        let mut square = Square::new()?;

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let y = Arc::new(variable!(vec![vec![3.0]]));
        let z = add.apply(&[&square.apply(&[&x])?, &square.apply(&[&y])?])?; // z = add(square(x), square(y))
        assert_eq!(z.tensor().data(), Tensor::new(vec![vec![13.0]]).data());

        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_eq!(x.grad(), Some(&Tensor::new(vec![vec![4.0]])));
            assert_eq!(y.grad(), Some(&Tensor::new(vec![vec![6.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf5() -> MlResult<()> {
        let mut add = Add::new()?;
        let mut mul = Mul::new()?;

        let a = Arc::new(variable!(vec![vec![3.0]]));
        let b = Arc::new(variable!(vec![vec![2.0]]));
        let c = Arc::new(variable!(vec![vec![1.0]]));

        let y = add.apply(&[&mul.apply(&[&a, &b])?, &c])?;

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_eq!(y.tensor(), &Tensor::new(vec![vec![7.0]]));
            assert_eq!(a.grad(), Some(&Tensor::new(vec![vec![2.0]])));
            assert_eq!(b.grad(), Some(&Tensor::new(vec![vec![3.0]])));
        }
        Ok(())
    }

    #[test]
    fn wtf6() -> MlResult<()> {
        // let mut pow = Pow::new()?;
        // pow.power = Some(3.0);
        
        todo!("Pow operator test is not implemented yet"); // Placeholder for actual test implementation
        
        // let x = Arc::new(variable!(vec![vec![2.0]]));
        // let y = pow.apply(&[&x])?; // y = x^3
        // 
        // #[cfg(feature = "enableBackpropagation")]
        // {
        //     y.backward()?; // dy/dx = 3x^2
        // 
        //     assert_eq!(y.tensor(), &Tensor::new(vec![vec![8.0]]));
        //     assert_eq!(x.grad(), Some(Tensor::new(vec![vec![12.0]])));
        // }
        // Ok(())
    }

    #[test]
    fn trigonometry_sin() -> MlResult<()> {
        let mut sin = Sin::new()?;

        let x = Arc::new(variable!(vec![vec![std::f32::consts::PI / 4.0]])); // 45도 (45 * 4 = 180)
        let y = sin.apply(&[&x])?;

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_tensor_eq(y.tensor(), &Tensor::new(vec![vec![std::f32::consts::FRAC_1_SQRT_2]]))?;
            assert_tensor_eq(x.grad().unwrap(), &Tensor::new(vec![vec![std::f32::consts::FRAC_1_SQRT_2]]))?;
        }
        Ok(())
    }
}
