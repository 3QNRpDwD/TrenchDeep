pub mod tensor;
pub mod backend;
pub mod nn;
pub mod optimizer;
pub mod loss;
pub mod trainer;
#[cfg(test)]
mod tests;

use crate::backend::BackendError;
use crate::loss::LossError;
use crate::optimizer::OptimError;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum TensorError {
    #[error("Invalid shape: expected {:?}, got {:?}", expected, got)]
    InvalidShape { expected: Vec<usize>, got: Vec<usize>, },
    #[error("Invalid data length: expected {}, got {}", expected, got)]
    InvalidDataLength { expected: usize, got: usize, },
    #[error("Invalid operation '{}': {}", op, reason)]
    InvalidOperation { op: &'static str, reason: String, },
    #[error("Invalid axis {} for tensor with shape {:?}", axis, shape)]
    InvalidAxis { axis: usize, shape: Vec<usize>, },
    #[error("Invalid dimensions for matrix multiplication: left shape {:?}, right shape {:?}", left_shape, right_shape)]
    MatrixMultiplicationError { left_shape: Vec<usize>, right_shape: Vec<usize>, },
    #[error("InvalidInputCount: expected {:?}, got {:?}", expected, got)]
    InvalidInputCount { expected: i32, got: usize },
    #[error("Empty tensor")]
    EmptyTensor,
}

#[derive(Error, Debug)]
pub enum MlError {
    #[error(transparent)]
    TensorError(#[from] TensorError),
    #[error(transparent)]
    LossError(#[from] LossError),
    #[error("{0}")]
    StringError(String),
    #[error(transparent)]
    BackendError(#[from] BackendError),
    #[error(transparent)]
    OptimError(#[from] OptimError),
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
    use std::ops::SubAssign;
    use crate::tensor::operators::{Add, Function, Mul, Square};
    use crate::tensor::{AutogradFunction, ComputationGraph, Tensor, TensorBase};
    use crate::{MlResult, scalar, var_input, var_with_label, variable};
    use crate::nn::{Parameter, Variable};
    use crate::tests::common::logging::setup_logging;

    fn assert_tensor_eq(tensor: &Tensor, expected_tensor: &Tensor) -> MlResult<()> {
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

    fn sphere_function(x: &Variable, y: &Variable) -> MlResult<Variable> {
        let mut square = Square::new()?;
        Ok(&square.apply(&[x])? + &square.apply(&[y])?)
    }

    fn matyas_function(x: &Variable, y: &Variable) -> MlResult<Variable> {
        let O_26 = variable!(vec![vec![0.26]]);
        let O_48 = variable!(vec![vec![0.48]]);

        let sphere = sphere_function(x, y)?;
        let t = x * y;                                    // x * y
        Ok(&(&O_26 * &sphere) - &(&O_48 * &t))            // 0.26 * sphere - 0.48 * x * y
    }

    fn goldstein_price_function(x: &Variable, y: &Variable) -> MlResult<Variable> {
        fn constant(value: f32) -> Variable {
            let scalar = Tensor::scalar(value);
            var_with_label!(scalar, &value.to_string())
        }

        let mut square = Square::new()?;
        let mut mul = Mul::new()?;

        let num_1   = constant(1.0);
        let num_2   = constant(2.0);
        let num_3   = constant(3.0);
        let num_6   = constant(6.0);
        let num_12  = constant(12.0);
        let neg_14  = constant(-14.0);
        let neg_32  = constant(-32.0);
        let neg_36  = constant(-36.0);

        // a = x + y + 1
        let a = &(x + y) + &num_1;

        // x^2, y^2
        let x_squared = square.apply(&[x])?;
        let y_squared = square.apply(&[y])?;

        // b = 19 - 14x + 3x^2 - 14y + 6xy + 3y^2
        let b = &(&(&(&(&constant(19.0) + &(&neg_14 * x))
            + &(&num_3 * &x_squared))
            + &(&neg_14 * y))
            + &(&num_6 * &(x * y)))
            + &(&num_3 * &y_squared);

        // first_part = 1 + a^2 * b
        let a_squared = square.apply(&[&a])?;
        let first_part = &num_1 + &(&a_squared * &b);

        // c = 2x - 3y
        let c = &(&num_2 * x) - &(&num_3 * y);

        // d = 18 - 32x + 12x^2 + 48y - 36xy + 27y^2
        let d = &(&(&(&(&constant(18.0) + &(&neg_32 * x))
            + &(&num_12 * &x_squared))
            + &(&constant(48.0) * y))
            + &(&neg_36 * &(x * y)))
            + &(&constant(27.0) * &y_squared);

        // second_part = 30 + c^2 * d
        let c_squared = square.apply(&[&c])?;
        let second_part = &constant(30.0) + &(&c_squared * &d);

        // output = first_part * second_part
        mul.apply_with_label(&[&first_part, &second_part], "output")
    }

    fn goldstein_price_function_v2(x: &Variable, y: &Variable) -> MlResult<Variable> {
        let c = |v: f32| variable!(vec![vec![v]]);
        let mut s = Square::new()?;
        let (x2, y2, xy) = (s.apply(&[x])?, s.apply(&[y])?, x * y);

        let f1 = &c(1.0) + &(&s.apply(&[&(x + y + &c(1.0))])? * &(
            &c(19.0) - &(x * &c(14.0)) + &(&x2 * &c(3.0)) - &(y * &c(14.0)) + &(&xy * &c(6.0)) + &(&y2 * &c(3.0))
        ));

        let f2 = &c(30.0) + &(&s.apply(&[&(&(x * &c(2.0)) - &(y * &c(3.0)))])? * &(
            &c(18.0) - &(x * &c(32.0)) + &(&x2 * &c(12.0)) + &(y * &c(48.0)) - &(&xy * &c(36.0)) + &(&y2 * &c(27.0))
        ));

        Ok(f1 * f2)
    }

    fn rosenbrock_function(x0: &Variable, x1: &Variable) -> MlResult<Variable> {
        let mut square = Square::new()?;
        let mut add = Add::new()?;

        let sq = square.apply(&[&x0])?;//x0^2
        // 100 * (x1 - x0^2)^2 + (1 - x0)^2
        let term1 = &variable!(vec![vec![100.0]]) * &square.apply(&[&(x1 - &sq)])?;
        let term2 = square.apply(&[&(&variable!(vec![vec![1.0]]) - x0)])?;
        add.apply_with_label(&[&term1, &term2], "output")
    }

    #[test]
    fn sphere() -> MlResult<()> {
        let x = var_input!(Tensor::new(vec![vec![1.0]]));
        let y = var_input!(Tensor::new(vec![vec![1.0]]));
        let z = sphere_function(&x, &y)?;
        #[cfg(feature = "enableBackward")]
        {
            z.backward()?;

            assert_tensor_eq(&x.grad(), &Tensor::new(vec![vec![2.0]]))?;
            assert_tensor_eq(&y.grad(), &Tensor::new(vec![vec![2.0]]))?;
        }
        Ok(())
    }

    #[test]
    fn matyas() -> MlResult<()> {
        let x = var_input!(Tensor::new(vec![vec![1.0]]));
        let y = var_input!(Tensor::new(vec![vec![1.0]]));
        let z = matyas_function(&x, &y)?;
        #[cfg(feature = "enableBackward")]
        z.backward()?;
        Ok(())
    }

    #[test]
    fn goldstein() -> MlResult<()> {
        let x = var_input!(Tensor::from_vec(vec![1.0], &[1,1])?);
        let y = var_input!(Tensor::from_vec(vec![1.0], &[1,1])?);
        let z = goldstein_price_function(&x, &y)?;
        #[cfg(feature = "enableBackward")]
        {
            z.backward()?;

            assert_tensor_eq(x.grad(), &Tensor::new(vec![vec![-5376.0]]))?;
            assert_tensor_eq(y.grad(), &Tensor::new(vec![vec![8064.0]]))?;
        }

        #[cfg(feature = "enableVisualization")]
        crate::tensor::VisualizationGraph::save_graph("graph/goldstein.dot").unwrap();
        Ok(())
    }

    #[test]
    fn goldstein_v2() -> MlResult<()> {
        let x = var_input!(Tensor::from_vec(vec![1.0], &[1, 1])?);
        let y = var_input!(Tensor::from_vec(vec![1.0], &[1, 1])?);
        let z = goldstein_price_function_v2(&x, &y)?;
        #[cfg(feature = "enableBackward")]
        {
            z.backward()?;

            assert_tensor_eq(x.grad(), &Tensor::new(vec![vec![-5376.0]]))?;
            assert_tensor_eq(y.grad(), &Tensor::new(vec![vec![8064.0]]))?;
        }
        Ok(())
    }

    #[test]
    fn rosenbrock() -> MlResult<()> {
        let x0 = var_with_label!(Tensor::from_vec(vec![0.0], &[1,1])?, "x0");
        let x1 = var_with_label!(Tensor::from_vec(vec![2.0], &[1,1])?, "x1");
        let y = rosenbrock_function(&x0, &x1)?;

        #[cfg(feature = "enableBackward")]
        {
            y.backward()?;

            assert_tensor_eq(x0.grad(), &Tensor::new(vec![vec![-2.0]]))?;
            assert_tensor_eq(x1.grad(), &Tensor::new(vec![vec![400.0]]))?;
        }

        #[cfg(feature = "enableVisualization")]
        {
            crate::tensor::VisualizationGraph::save_graph("graph/rosenbrock.dot").unwrap();
            crate::tensor::VisualizationGraph::render_to_svg("graph/rosenbrock.svg").unwrap();
        }
        Ok(())
    }

    #[ignore = "너무 오래걸려서 무시함"]
    #[test]
    #[cfg(feature = "enableBackward")]
    fn rosenbrock_gradient_descent_function() -> MlResult<()> {
        let mut x0 = var_input!(Tensor::new(vec![vec![0.0]]));
        let mut x1 = var_input!(Tensor::new(vec![vec![2.0]]));
        let iter: usize = 1000;
        let learning_rate = Tensor::scalar(0.001);

        for i in 0..iter { // 0부터
            ComputationGraph::reset_graph();
            let y = rosenbrock_function(&x0, &x1)?;
            y.backward()?;
            
            if i % 1 == 0 {
                println!(
                    "iter - {}\n\
            [ x0.tensor: {:?}, x0.grad: {:?} ]\n\
            [ x1.tensor: {:?}, x1.grad: {:?} ]"
                    , i, x0.tensor(), x0.grad(), x1.tensor(), x1.grad()
                );
            }
            
            // 파라미터 갱신
            x0 -= Variable::new(x0.grad() * &learning_rate);
            x1 -= Variable::new(x1.grad() * &learning_rate);
        }
        Ok(())
    }
}
