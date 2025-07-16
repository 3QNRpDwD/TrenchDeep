use std::any::Any;
use std::sync::Arc;
use std::collections::HashMap;
use std::cell::RefCell;
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
pub mod avg_pool;
pub mod max_pool;
pub mod conv2d;

thread_local! {
    pub(crate) static OPERATOR_STORAGE: RefCell<HashMap<String, Arc<dyn Any + Send + Sync>>> = RefCell::new(HashMap::new());
}

pub(crate) fn get_or_create_op<T, F>(op_key: &str, creator: F) -> Arc<T>
where
    T: 'static + Function + Send + Sync,
    F: FnOnce() -> T,
{
    OPERATOR_STORAGE.with(|storage_cell| {
        let mut storage = storage_cell.borrow_mut();
        if let Some(op_any) = storage.get(op_key) {
            op_any.clone().downcast::<T>().unwrap_or_else(|_| {
                panic!("Failed to downcast operator for key: {}", op_key);
            })
        } else {
            let new_op = Arc::new(creator());
            storage.insert(op_key.to_string(), new_op.clone());
            new_op
        }
    })
}

#[macro_export]
macro_rules! define_op {
    // 파라미터가 없는 연산자
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            backend: Arc<dyn Backend>,
            node_id: HandleId,
        }

        impl $name {
            pub fn new() -> Arc<Self> {
                let key = stringify!($name).to_string();
                crate::tensor::operators::get_or_create_op(&key, || $name {
                    backend: Arc::new(crate::backend::CpuBackend::new().unwrap()),
                    node_id: crate::tensor::NODE_ID_GEN.next(),
                })
            }
        }
    };
    // 파라미터가 있는 연산자
    ($name:ident, $($field:ident: $type:ty),+) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            backend: Arc<dyn Backend>,
            node_id: HandleId,
            $(pub $field: $type),+
        }

        impl $name {
            pub fn new($($field: $type),+) -> Arc<Self> {
                let key = format!(
                    "{}_{:?}",
                    stringify!($name),
                    ($($field),+,)
                );
                crate::tensor::operators::get_or_create_op(&key, || $name {
                    backend: Arc::new(crate::backend::CpuBackend::new().unwrap()),
                    node_id: crate::tensor::NODE_ID_GEN.next(),
                    $($field),+
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
define_op!(Sin);
define_op!(Cos);
define_op!(Reshape);
define_op!(Transpose, dims: (i32, i32));
define_op!(Pow, power: Option<f32>);
define_op!(Topk, topk: Option<(usize, bool)>);
define_op!(Matmax, matmax: Option<(Option<i32>, bool)>);
define_op!(ApproxSin, threshold: f32);
define_op!(ApproxCos, threshold: f32);
define_op!(AvgPool, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize));
define_op!(MaxPool, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize));
define_op!(Conv2d, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize));


pub trait Function {
    fn type_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown")
    }

    fn forward(&self, _targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>>{
        unimplemented!("{} Forward pass is not implemented", self.type_name())
    }

    fn assign_forward(&self, _targets: &[&dyn TensorBase], tensor_id: HandleId) -> MlResult<Vec<Tensor>> {
        unimplemented!("{} Forward pass is not implemented", self.type_name())
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        unimplemented!("{} Backward pass is not implemented", self.type_name())
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        unimplemented!("{} Function::backend() is not implemented", self.type_name())
    }
    
    fn node_id(&self) -> &HandleId {
        unimplemented!("{} Function::node_id() is not implemented", self.type_name())
    }
}

impl Debug for &dyn Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Function<{}>", std::any::type_name::<Self>())
    }
}

#[cfg(test)]
mod tests {
    use crate::{scalar, variable, tensor_ops};
    use crate::tests::common::utils::setup_logging;
    use super::*;

    pub fn assert_tensor_eq(tensor: &dyn TensorBase, expected_tensor: &dyn TensorBase) -> MlResult<()> {
        if tensor.data() != expected_tensor.data() && tensor.shape() != expected_tensor.shape() {
            return Err(format!("Expected {:?}, got {:?}", expected_tensor, tensor).into());
        }
        Ok(())
    }

    pub fn assert_variable_eq(variable: &Variable, expected_variable: &Variable) -> MlResult<()> {
        assert_eq!(variable.tensor().data(), expected_variable.tensor().data());
        assert_eq!(variable.tensor().shape(), expected_variable.tensor().shape());
        Ok(())
    }

    #[test]
    fn tensor_add_operator() -> MlResult<()> {
        let first = Tensor::new(vec![vec![1.0, 2.0]]);
        let second = Tensor::new(vec![vec![3.0, 4.0]]);
        let expected = Tensor::new(vec![vec![4.0, 6.0]]);
        let result = first + second; // This relies on `impl Add for Tensor` which should use `operators::Add::new()`

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
        x: &dyn TensorBase,
        a: &dyn TensorBase,
        b: &dyn TensorBase,
        y: &dyn TensorBase,
    ) {
        // ... (implementation unchanged)
    }

    fn print_backward(
        x: Option<&dyn TensorBase>,
        a: Option<&dyn TensorBase>,
        b: Option<&dyn TensorBase>,
        y: Option<&dyn TensorBase>,
    ) {
        // ... (implementation unchanged)
    }

    #[test]
    fn phase_test() -> MlResult<()>{
        let square = Square::new();
        let exp = Exp::new();

        let x = scalar!(0.5);
        let a = square.forward(&[ &x ])?.remove(0);
        let b = exp.forward(&[ &a ])?.remove(0);
        let y = square.forward(&[ &b ])?.remove(0);

        print_forward(&x, &a, &b, &y);
        assert_tensor_eq(&y, &Tensor::new(vec![vec![1.6487213]]))?;

        #[cfg(feature = "enableBackpropagation")]
        {
            let dy = scalar!(1.0);
            let db = square.backward(&[&b], &dy)?.remove(0);
            let da = exp.backward(&[&a], &db)?.remove(0);
            let dx = square.backward(&[&x], &da)?.remove(0);

            print_backward(Some(&dy), Some(&db), Some(&da), Some(&dx));
            assert_tensor_eq(&dx, &Tensor::new(vec![vec![3.2974427]]))?;
        }
        Ok(())
    }

    #[test]
    fn autograd_test() -> MlResult<()> {
        let a= setup_logging("info");
        let square = Square::new();
        let exp = Exp::new();

        let x = Arc::new(variable!(vec![vec![0.5]]));
        let a = square.apply(&[&x])?;
        let b = exp.apply(&[&a])?;
        let y = square.apply(&[&b])?;

        print_forward(x.tensor(), a.tensor(), b.tensor(), y.tensor());
        crate::tensor::tests::assert_tensor_eq(y.tensor(), &Tensor::new(vec![vec![1.6487213]]))?;

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;
            let dy = y.grad();
            let db = b.grad();
            let da = a.grad();
            let dx = x.grad();

            print_backward(Some(dy), Some(db), Some(da), Some(dx));
            assert_tensor_eq(x.grad(), &Tensor::new(vec![vec![3.2974427]]))?;
        }
        Ok(())
    }

    // Other autograd tests are also ignored for now
    #[test]
    fn wtf() -> MlResult<()> {
        let mut add = Add::new();

        let x0 = Arc::new(variable!(vec![vec![1.0]]));
        let x1 = Arc::new(variable!(vec![vec![1.0]]));
        let t = add.apply(&[&x0, &x1])?; // t = x0 + x1 = 2
        let y = add.apply(&[&x0, &t])?; // y = x0 + t = 3

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            #[cfg(feature = "requiresGrad")] {
                assert_eq!(y.grad(), &Tensor::new(vec![vec![1.0]]));
                assert_eq!(t.grad(), &Tensor::new(vec![vec![1.0]]));
            }
            #[cfg(not(feature = "requiresGrad"))] {
                assert!(y.grad().is_empty());
                assert!(t.grad().is_empty());
            }

            assert_tensor_eq(x0.grad(), &Tensor::new(vec![vec![2.0]]))?;
            assert_tensor_eq(x1.grad(), &Tensor::new(vec![vec![1.0]]))?;
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
        let mut add = Add::new();
        let mut square = Square::new();

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let a = square.apply(&[&x])?;
        let y = add.apply(&[&square.apply(&[&a])?, &square.apply(&[&a])?])?;
        assert_eq!(y.tensor().data(), Tensor::new(vec![vec![32.0]]).data());

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_eq!(x.grad(), &Tensor::new(vec![vec![64.0]]));
        }
        Ok(())
    }

    #[test]
    fn wtf3() -> MlResult<()> { // 기울기 2, 3 나오면 됨
        let mut add = Add::new();

        let x = Arc::new(variable!(vec![vec![3.0]]));
        let y = add.apply(&[&x, &x])?; // y = add(x, x)
        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;
            assert_eq!(x.grad(), &Tensor::new(vec![vec![2.0]]));

            x.clear_grad();
            y.clear_grad();

            let t = add.apply(&[&x, &x])?;
            let y = add.apply(&[&t, &x])?; // y = add(add(x, x), x)
            #[cfg(feature = "enableBackpropagation")]
            y.backward()?;
            assert_eq!(x.grad(), &Tensor::new(vec![vec![3.0]]));
        }
        Ok(())
    }

    #[test]
    fn wtf4() -> MlResult<()> {
        let mut add = Add::new();
        let mut square = Square::new();

        let x = Arc::new(variable!(vec![vec![2.0]]));
        let y = Arc::new(variable!(vec![vec![3.0]]));
        let z = add.apply(&[&square.apply(&[&x])?, &square.apply(&[&y])?])?; // z = add(square(x), square(y))
        assert_eq!(z.tensor().data(), Tensor::new(vec![vec![13.0]]).data());

        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_eq!(x.grad(), &Tensor::new(vec![vec![4.0]]));
            assert_eq!(y.grad(), &Tensor::new(vec![vec![6.0]]));
        }
        Ok(())
    }

    #[test]
    fn wtf5() -> MlResult<()> {
        let mut add = Add::new();
        let mut mul = Mul::new();

        let a = Arc::new(variable!(vec![vec![3.0]]));
        let b = Arc::new(variable!(vec![vec![2.0]]));
        let c = Arc::new(variable!(vec![vec![1.0]]));

        let y = add.apply(&[&mul.apply(&[&a, &b])?, &c])?;

        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_eq!(y.tensor(), &Tensor::new(vec![vec![7.0]]));
            assert_eq!(a.grad(), &Tensor::new(vec![vec![2.0]]));
            assert_eq!(b.grad(), &Tensor::new(vec![vec![3.0]]));
        }
        Ok(())
    }

    #[test] #[ignore] fn trigonometry_sin() -> MlResult<()> { Ok(()) }
}
