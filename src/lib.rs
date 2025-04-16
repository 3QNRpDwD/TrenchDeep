use std::fmt::{Display, Formatter};
use crate::tensor::TensorError;

pub mod tensor;
pub mod backend;
pub mod nn;
pub mod optimizer;
pub mod loss;


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
mod benchmark {
    use std::sync::{Arc};
    use crate::{MlResult, scalar_ops, variable};
    use crate::tensor::{Tensor, TensorBase, Variable};
    use crate::tensor::creation::AutogradFunction;
    use crate::tensor::operators::{Add, Function, Mul, Pow, Square, Sub};

    fn assert_tensor_eq(tensor: &Tensor<f32>, expected_tensor: &Tensor<f32>) -> MlResult<()> {
        if tensor.shape() != expected_tensor.shape() {
            return Err("Shape mismatch".into());
        }

        let tensor_data = tensor.data();
        let expected_data = expected_tensor.data();

        for (t, e) in tensor_data.iter().zip(expected_data.iter()) {
            if (t - e).abs() > 1e-6 {
                return Err("Data mismatch".into());
            }
        }

        Ok(())
    }

    fn sphere_function(x: &Arc<Variable<f32>>, y: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        let mut pow = Pow::new()?;
        let add = Add::new()?;
        pow.power = Some(2.0);

        add.apply(&[
            &pow.apply(&[x])?,
            &pow.apply(&[y])?]
        )
    }

    fn matyas_function(x: &Arc<Variable<f32>>, y: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        let sub = Sub::new()?;
        let mul = Mul::new()?;
        let O_26 = Arc::new(variable!(vec![vec![0.26]]));
        let O_48 = Arc::new(variable!(vec![vec![0.48]]));

        let sphere = sphere_function(x, y)?;
        sub.apply(&[                   // (0.26 * sphere) - (0.48 * x * y)
            &mul.apply(&[&O_26, &sphere])?,                     // 0.26 * sphere
            &mul.apply(&[&O_48, &mul.apply(&[x, y])?])?  // 0.48 * x * y
        ])
    }

    fn goldstein_price_function(x: &Arc<Variable<f32>>, y: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        // Helper function to create constant variables
        fn constant(value: f32) -> Arc<Variable<f32>> {
            Arc::new(variable!(vec![vec![value]]))
        }

        let add = Add::new()?;
        let square = Square::new()?;
        let mul = Mul::new()?;
        let sub = Sub::new()?;

        // Define constants
        let num_1   = constant(1.0);
        let num_2   = constant(2.0);
        let num_3   = constant(3.0);
        let num_6   = constant(6.0);
        let num_12  = constant(12.0);
        let neg_14  = constant(-14.0);
        let neg_32  = constant(-32.0);
        let neg_36  = constant(-36.0);

        // Compute a = x + y + 1
        let a =
            add.apply(&[
                &add.apply(&[x, y])?,
                &num_1
            ])?;

        // Compute x squared and y squared
        let x_squared = square.apply(&[x])?;
        let y_squared = square.apply(&[y])?;
        // Compute b = (((((19 - 14x) + 3x^2) - 14y) + 6xy) + 3y^2)
        let term2_b = mul.apply(&[&neg_14, x])?;
        let term3_b = mul.apply(&[&num_3, &x_squared])?;
        let term4_b = mul.apply(&[&neg_14, y])?;
        let term5_b = mul.apply(&[&num_6, &mul.apply(&[x, y])?])?;
        let term6_b = mul.apply(&[&num_3, &y_squared])?;

        let b =
            add.apply(&[
                &add.apply(&[
                    &add.apply(&[
                        &add.apply(&[
                            &add.apply(&[&constant(19.0), &term2_b])?,
                            &term3_b])?,
                        &term4_b])?,
                    &term5_b])? ,
                &term6_b
            ])?; // (((((19 - 14x) + 3x^2) - 14y) + 6xy) + 3y^2)

        // Compute first part: 1 + (a^2 * b)
        let a_squared   = square.apply(&[&a])?;
        let a_squared_b = mul.apply(&[&a_squared, &b])?;
        let first_part  = add.apply(&[&num_1, &a_squared_b])?;

        // Compute c = 2x - 3y
        let two_x   = mul.apply(&[&num_2, x])?;
        let three_y = mul.apply(&[&num_3, y])?;
        let c       = sub.apply(&[&two_x, &three_y])?;

        // Compute d = 18 - 32x + 12x^2 + 48y - 36xy + 27y^2
        let term2_d = mul.apply(&[&neg_32, x])?;
        let term3_d = mul.apply(&[&num_12, &x_squared])?;
        let term4_d = mul.apply(&[&constant(48.0), y])?;
        let term5_d = mul.apply(&[&neg_36, &mul.apply(&[x, y])?])?;
        let term6_d = mul.apply(&[&constant(27.0), &y_squared])?;

        let d =
            add.apply(&[
                &add.apply(&[
                    &add.apply(&[
                        &add.apply(&[
                            &add.apply(&[&constant(18.0), &term2_d])?,
                            &term3_d])?,
                        &term4_d])?,
                    &term5_d])? ,
                &term6_d])?; // 18 - 32x + 12x^2 + 48y - 36xy + 27y^2

        // Compute second part: 30 + c^2 * d
        let c_squared   = square.apply(&[&c])?;
        let c_squared_d = mul.apply(&[&c_squared, &d])?;
        let second_part = add.apply(&[&constant(30.0), &c_squared_d])?;

        // Compute final function value
        mul.apply(&[&first_part, &second_part])
    }

    fn rosenbrock_function(x0: &Arc<Variable<f32>>, x1: &Arc<Variable<f32>>) -> MlResult<Arc<Variable<f32>>> {
        let sub = Sub::new()?;
        let add = Add::new()?;
        let square = Square::new()?;
        let mul = Mul::new()?;

        add.apply(&[
            &mul.apply(&[
                &Arc::new(variable!(vec![vec![100.0]])),
                &square.apply(&[
                    &sub.apply(&[
                        &x1,
                        &square.apply(&[&x0])?])?
                ])?
            ])?,
            &square.apply(&[
                &sub.apply(&[
                    &Arc::new(variable!(vec![vec![1.0]])),
                    &x0
                ])?
            ])?
        ])
    }

    #[test]
    fn sphere() -> MlResult<()> {
        let x = Arc::new(variable!(vec![vec![1.0]]));
        let y = Arc::new(variable!(vec![vec![1.0]]));
        let z = sphere_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
            assert_tensor_eq(&y.grad().unwrap(), &Tensor::new(vec![vec![2.0]]))?;
        }
        Ok(())
    }

    #[test]
    fn matyas() -> MlResult<()> {
        let x = Arc::new(variable!(vec![vec![1.0]]));
        let y = Arc::new(variable!(vec![vec![1.0]]));
        let z = matyas_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            #[cfg(feature = "debugging")]
            {
                println!("matyas - x.grad: {:?}", x.grad());
                println!("matyas - y.grad: {:?}", y.grad());
            }
        }
        Ok(())
    }

    #[test]
    fn goldstein() -> MlResult<()> {
        let x = Arc::new(variable!(vec![vec![1.0]]));
        let y = Arc::new(variable!(vec![vec![1.0]]));
        let z = goldstein_price_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]{
            z.backward()?;

            assert_tensor_eq(&x.grad().unwrap(), &Tensor::new(vec![vec![-5376.0]]))?;
            assert_tensor_eq(&y.grad().unwrap(), &Tensor::new(vec![vec![8064.0]]))?;
        }
        Ok(())
    }

    #[test]
    fn rosenbrock() -> MlResult<()> {
        let x0 = Arc::new(variable!(vec![vec![0.0]]));
        let x1 = Arc::new(variable!(vec![vec![2.0]]));

        let y = rosenbrock_function(&x0, &x1)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_tensor_eq(&x0.grad().unwrap(), &Tensor::new(vec![vec![-2.0]]))?;
            assert_tensor_eq(&x1.grad().unwrap(), &Tensor::new(vec![vec![400.0]]))?;
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "enableBackpropagation")]
    fn rosenbrock_gradient_descent_function() -> MlResult<()> {
        let sub = Sub::new()?;
        let mul = Mul::new()?;
        let mut x0 = Arc::new(variable!(vec![vec![0.0]]));
        let mut x1 = Arc::new(variable!(vec![vec![2.0]]));
        let iter: usize = 50000;
        let learning_rate = Tensor::new(vec![vec![0.001]]);

        for i in 0..iter { // 0부터
            crate::tensor::creation::COMPUTATION_GRAPH.with(|graph| {
                let mut graph_guard = graph.lock().unwrap();
                *graph_guard = crate::tensor::ComputationGraph::new();
            });
            // 계산그래프가 초기화되지 않고, 계속해서 텐서가 쌓이던것이 성능 하락의 주요 원인이었으며,
            // 매번 초기화하는 임시방편을 적용했으나, 이는 바람직한 해결방법이 아니며, 여전히 많은 리소스가 낭비되고 있는것으로 보임.
            // 기존의 계산그래프를 수정 가능하도록 변경하는것이 필요함
            // 단 임시방편이더라도 기존의 코드보다 훨신 성능이 나아짐. 5000개를 500ms 미만으로 계산가능하게 되었음.
            // 200개를 22초만에 계산하던 기존의 버전에서 5000개를 500ms 미만으로 계산가능한것은 엄청난 발전임.

            let y = rosenbrock_function(&x0, &x1)?;
            y.backward()?;

            if i == 50000-1 {
                println!(
                    "iter - {}\n\
                    [ x0.tensor: {:?}, x0.grad: {:?} ]\n\
                    [ x1.tensor: {:?}, x1.grad: {:?} ]"
                    , i, x0.tensor(), x0.grad(), x1.tensor(), x1.grad()
                );
            }

            let x0_mul_lr = mul.forward(&[&x0.grad().unwrap(), &learning_rate])?.remove(0);
            let x1_mul_lr = mul.forward(&[&x1.grad().unwrap(), &learning_rate])?.remove(0);
            x0 = Arc::new(Variable::new(sub.forward(&[x0.tensor(), &x0_mul_lr])?.remove(0)));
            x1 = Arc::new(Variable::new(sub.forward(&[x1.tensor(), &x1_mul_lr])?.remove(0)));
            // 현재 텐서의 불변성 유지 때문에 기존 텐서를 수정하는것이 아닌, 새로 생성하기 때문에,
            // 계산 그래프가 불필요하게 거대해지고, 연산시간이 오래걸리는 문제가 발생함.
            // 이는 Mutex 를 도입해서 멀티스레딩과 함께 텐서를 수정하는 메서드를 추가하는 해결책이 필요해보임.
            // 또한 기존의 계산그래프를 최적화해서 내부의 값을 수정하거나, 수정된 노드가 등록될 경우에 이전 노드를 삭제하는등의 방안 마련이 시급해보임.
            // 위와같이 생각했었지만. 동적그래프 기반의 역전파 구조가 잘못된 것이었음.
            // 결국 신경망은 한번 구현된 이후에 계속해서 데이터를 흘려보내기 때문에, 매 계산마다 노드를 추가할 필요가 없으며,
            // 계산 그래프에서는 텐서를 같이 저장하는것이 아닌, 계산 순서만 저장하는 방식으로 전환하는것이 바람직해보임.
            // 따라서 이러한 방향을 적용할 경우, apply 함수는 더이상 텐서를 입력받지 않을것으로 보이며, 내부적으로 계산그래프를 생성 후,
            // 입력 텐서를 외부에서 흘려넣는 방식이 될것이라고 생각됨.
            // 이는 그래프 구조체를 통해서 구현될지, 출력 텐서를 통해서 구현될지 아직 모호함. 하지만 정적 계산 그래프를 사용하는것은 확정된 사안임.
        }

        Ok(())
    }
}
