pub mod tensor;
pub mod backend;
pub mod nn;
pub mod optimizer;
pub mod loss;
pub mod tests;

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
    use crate::tensor::operators::{Add, Function, Mul, Square, Sub};
    use crate::tensor::{AutogradFunction, ComputationGraph, Tensor, TensorBase, Variable};
    use crate::{MlResult, scalar, var_input, var_with_label, variable};
    use std::sync::Arc;

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

    fn sphere_function(x: &Arc<Variable>, y: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        let mut square = Square::new()?;
        let mut add = Add::new()?;

        add.apply(&[
            &square.apply(&[x])?,
            &square.apply(&[y])?]
        )
    }

    fn matyas_function(x: &Arc<Variable>, y: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        let mut sub = Sub::new()?;
        let mut mul = Mul::new()?;
        let O_26 = Arc::new(variable!(vec![vec![0.26]]));
        let O_48 = Arc::new(variable!(vec![vec![0.48]]));

        let sphere = sphere_function(x, y)?;
        let t = mul.apply(&[x, y])?;
        sub.apply(&[                   // (0.26 * sphere) - (0.48 * x * y)
            &mul.apply(&[&O_26, &sphere])?,                     // 0.26 * sphere
            &mul.apply(&[&O_48, &t])?  // 0.48 * x * y
        ])
    }

    fn goldstein_price_function(x: &Arc<Variable>, y: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        // Helper function to create constant variables
        fn constant(value: f32) -> Arc<Variable> {
            let scalar = Tensor::scalar(value);
            var_with_label!(scalar, &value.to_string())
        }

        let mut add = Add::new()?;
        let mut square = Square::new()?;
        let mut mul = Mul::new()?;
        let mut sub = Sub::new()?;

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
        let tt = add.apply(&[x, y])?;
        let a =
            add.apply(&[
                &tt,
                &num_1
            ])?;

        // Compute x squared and y squared
        let x_squared = square.apply(&[x])?;
        let y_squared = square.apply(&[y])?;
        // Compute b = (((((19 - 14x) + 3x^2) - 14y) + 6xy) + 3y^2)
        let term2_b = mul.apply(&[&neg_14, x])?;
        let term3_b = mul.apply(&[&num_3, &x_squared])?;
        let term4_b = mul.apply(&[&neg_14, y])?;
        let tb = mul.apply(&[x, y])?;
        let term5_b = mul.apply(&[&num_6, &tb])?;
        let term6_b = mul.apply(&[&num_3, &y_squared])?;

        let b = {
                let t1 = add.apply(&[&constant(19.0), &term2_b])?;      // 19 - 14x
                let t2 = add.apply(&[&t1, &term3_b])?;                  // + 3x^2
                let t3 = add.apply(&[&t2, &term4_b])?;                  // - 14y
                let t4 = add.apply(&[&t3, &term5_b])?;                  // + 6xy
                add.apply(&[&t4, &term6_b])?                            // + 3y^2
        }; // (((((19 - 14x) + 3x^2) - 14y) + 6xy) + 3y^2)

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
        let tb = mul.apply(&[x, y])?;
        let term5_d = mul.apply(&[&neg_36, &tb])?;
        let term6_d = mul.apply(&[&constant(27.0), &y_squared])?;

let d = {
                // 18 - 32x + 12x^2 + 48y - 36xy + 27y^2
                let t1 = add.apply(&[&constant(18.0), &term2_d])?;      // 18 - 32x
                let t2 = add.apply(&[&t1, &term3_d])?;                  // + 12x^2
                let t3 = add.apply(&[&t2, &term4_d])?;                  // + 48y
                let t4 = add.apply(&[&t3, &term5_d])?;                  // - 36xy
                add.apply(&[&t4, &term6_d])?                            // + 27y^2
            };

        // Compute second part: 30 + c^2 * d
        let c_squared   = square.apply(&[&c])?;
        let c_squared_d = mul.apply(&[&c_squared, &d])?;
        let second_part = add.apply(&[&constant(30.0), &c_squared_d])?;

        // Compute final function value
        mul.apply_with_label(&[&first_part, &second_part], "output")
    }

    fn rosenbrock_function(x0: &Arc<Variable>, x1: &Arc<Variable>) -> MlResult<Arc<Variable>> {
        let mut sub = Sub::new()?;
        let mut add = Add::new()?;
        let mut square = Square::new()?;
        let mut mul = Mul::new()?;

        let sq = square.apply(&[&x0])?;
        add.apply_with_label(&[
            &mul.apply(&[
                &Arc::new(variable!(vec![vec![100.0]])),
                &square.apply(&[
                    &sub.apply(&[
                        &x1,
                        &sq
                    ])?
                ])?
            ])?,
            &square.apply(&[
                &sub.apply(&[
                    &Arc::new(variable!(vec![vec![1.0]])),
                    &x0
                ])?
            ])?
        ], "output")
    }

    #[test]
    fn sphere() -> MlResult<()> {
        let x = var_input!(Tensor::new(vec![vec![1.0]]));
        let y = var_input!(Tensor::new(vec![vec![1.0]]));
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
        let x = var_input!(Tensor::new(vec![vec![1.0]]));
        let y = var_input!(Tensor::new(vec![vec![1.0]]));
        let z = matyas_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        z.backward()?;
        Ok(())
    }

    #[test]
    fn goldstein() -> MlResult<()> {
        let x = var_input!(Tensor::from_vec(vec![1.0], &[1,1])?);
        let y = var_input!(Tensor::from_vec(vec![1.0], &[1,1])?);
        let z = goldstein_price_function(&x, &y)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            z.backward()?;

            assert_tensor_eq(x.grad().unwrap(), &Tensor::new(vec![vec![-5376.0]]))?;
            assert_tensor_eq(y.grad().unwrap(), &Tensor::new(vec![vec![8064.0]]))?;
        }

        #[cfg(feature = "enableVisualization")]
        crate::tensor::VisualizationGraph::save_graph("graph/goldstein.dot").unwrap();
        Ok(())
    }

    #[test]
    fn rosenbrock() -> MlResult<()> {
        let x0 = var_input!(Tensor::from_vec(vec![0.0], &[1,1])?);
        let x1 = var_input!(Tensor::from_vec(vec![2.0], &[1,1])?);

        let y = rosenbrock_function(&x0, &x1)?;
        #[cfg(feature = "enableBackpropagation")]
        {
            y.backward()?;

            assert_tensor_eq(x0.grad().unwrap(), &Tensor::new(vec![vec![-2.0]]))?;
            assert_tensor_eq(x1.grad().unwrap(), &Tensor::new(vec![vec![400.0]]))?;
        }

        #[cfg(feature = "enableVisualization")]
        crate::tensor::VisualizationGraph::save_graph("graph/rosenbrock.dot").unwrap();
        Ok(())
    }

    #[test]
    #[cfg(feature = "enableBackpropagation")]
    fn rosenbrock_gradient_descent_function() -> MlResult<()> {
        let x0 = var_input!(Tensor::new(vec![vec![0.0]]));
        let x1 = var_input!(Tensor::new(vec![vec![2.0]]));
        let iter: usize = 1000;
        let learning_rate = Tensor::scalar(0.001);

        for i in 0..iter { // 0부터
            ComputationGraph::reset_graph();
            let y = rosenbrock_function(&x0, &x1)?;
            y.backward()?;
            
            // if i % 1 == 0 {
            //     println!(
            //         "iter - {}\n\
            // [ x0.tensor: {:?}, x0.grad: {:?} ]\n\
            // [ x1.tensor: {:?}, x1.grad: {:?} ]"
            //         , i, x0.tensor(), x0.grad(), x1.tensor(), x1.grad()
            //     );
            // }
            
            //파라미터 갱신
            x0.sub_tensor(x0.grad().unwrap() * &learning_rate)?;
            x1.sub_tensor( x1.grad().unwrap() * &learning_rate)?;
        }
        Ok(())
    }
}
