use super::*;
use crate::nn::Variable;
use crate::tensor::AutogradFunction;
use crate::tensor::broadcast::{broadcast_offsets, broadcast_shape, reduce_to_shape};

impl Function for Add {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Add)
    }
    
    /// Adds two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to add to the current tensor
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise addition
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let first_shape = first_target.shape();
        let second_shape = second_target.shape();

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Add::forward] {} + {}",
            crate::tensor::operators::debug::summary("lhs", first_target),
            crate::tensor::operators::debug::summary("rhs", second_target)
        );

        let result = if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting (hot path)
            #[cfg(feature = "debugging")]
            tracing::debug!("[Add::forward] broadcast path: [{},{}] + [{}]", first_shape[0], first_shape[1], second_shape[0]);

            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_target.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_target.data()[i * features + j] + second_target.data()[j];
                }
            }

            Ok(vec![GlobalTensor::from_vec(data, first_shape)?])
        } else if first_shape == second_shape {
            Ok(vec![GlobalTensor::from_vec(
                self.backend().add(first_target.data(), second_target.data()),
                first_target.shape(),
            )?])
        } else {
            // 범용 브로드캐스트 경로
            let out_shape = broadcast_shape(first_shape, second_shape)?;
            let a = first_target.data();
            let b = second_target.data();
            let offsets = broadcast_offsets(first_shape, second_shape, &out_shape);
            let data: Vec<f32> = offsets.iter().map(|&(ao, bo)| a[ao] + b[bo]).collect();
            Ok(vec![GlobalTensor::from_vec(data, &out_shape)?])
        };

        #[cfg(feature = "debugging")]
        if let Ok(ref r) = result {
            tracing::debug!("[Add::forward] → {}", crate::tensor::operators::debug::summary_raw("out", &r[0].data, &r[0].shape));
        }

        result
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: NodeId) -> MlResult<Vec<Tensor>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let first_shape = first_target.shape();
        let second_shape = second_target.shape();

        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_target.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_target.data()[i * features + j] + second_target.data()[j];
                }
            }

            return Ok(vec![Tensor::with_id(data, first_shape, node_id)?])
        }

        if first_shape == second_shape {
            return Ok(vec![Tensor::with_id(
                self.backend().add(first_target.data(), second_target.data()),
                first_target.shape(),
                node_id,
            )?]);
        }

        // 범용 브로드캐스트 경로
        let out_shape = broadcast_shape(first_shape, second_shape)?;
        let a = first_target.data();
        let b = second_target.data();
        let offsets = broadcast_offsets(first_shape, second_shape, &out_shape);
        let data: Vec<f32> = offsets.iter().map(|&(ao, bo)| a[ao] + b[bo]).collect();
        Ok(vec![Tensor::with_id(data, &out_shape, node_id)?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let first_shape = targets[0].shape();
        let second_shape = targets[1].shape();

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Add::backward] lhs_shape={:?} rhs_shape={:?}  {}",
            first_shape, second_shape,
            crate::tensor::operators::debug::summary("grad_in", grad)
        );

        // Broadcasting 케이스: [M, N] + [N] → bias grad는 batch 차원으로 합산 (hot path)
        let result = if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            #[cfg(feature = "debugging")]
            tracing::debug!("[Add::backward] broadcast path: summing bias grad over batch dim");

            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let grad_data = grad.data();
            let matrix_grad = GlobalTensor::from_vec(grad_data.to_vec(), first_shape)?;
            let mut bias_grad = vec![0.0f32; features];
            for i in 0..batch_size {
                for j in 0..features {
                    bias_grad[j] += grad_data[i * features + j];
                }
            }
            Ok(vec![matrix_grad, GlobalTensor::from_vec(bias_grad, second_shape)?])
        } else if first_shape == second_shape {
            let gt = GlobalTensor { data: grad.data().to_vec(), shape: grad.shape().to_vec(), dirty: false };
            Ok(vec![gt.clone(), gt])
        } else {
            // 범용 브로드캐스트 backward: grad 를 각 피연산자 원본 shape 로 축소
            let grad_shape = grad.shape();
            let grad_data = grad.data();
            let grad_a = reduce_to_shape(grad_data, grad_shape, first_shape);
            let grad_b = reduce_to_shape(grad_data, grad_shape, second_shape);
            Ok(vec![
                GlobalTensor::from_vec(grad_a, first_shape)?,
                GlobalTensor::from_vec(grad_b, second_shape)?,
            ])
        };

        #[cfg(feature = "debugging")]
        if let Ok(ref r) = result {
            for (i, g) in r.iter().enumerate() {
                crate::tensor::operators::debug::stats_raw(&format!("  └─ grad_out[{}]", i), &g.data, &g.shape);
            }
        }

        result
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    
    fn node_id(&self) -> &NodeId { &self.node_id }
}

/// Add trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to add to self
///
/// # Returns
/// A new tensor containing the element-wise sum
///
/// # Broadcasting
/// * Supports broadcasting when adding a 1D tensor to each row of a 2D tensor
impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().forward(&[&self, &other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().forward(&[&self, other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Self::Output {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().forward(&[self, other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Self::Output {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().forward(&[self, &other]).unwrap().remove(0))
            .to_id().unwrap()
    }
}

impl std::ops::Add<&dyn TensorBase> for &dyn TensorBase {
    type Output = GlobalTensor<f32>;

    fn add(self, other: &dyn TensorBase) -> Self::Output {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

/// AddAssign trait implementation for Tensor
impl std::ops::AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().assign_forward(&[self, &other], self.id()).unwrap().remove(0));
    }
}

impl std::ops::AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        Add::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Add").unwrap().assign_forward(&[self, other], self.id()).unwrap().remove(0));
    }
}

// Variable operator overloading (graph-tracked)
impl std::ops::Add<&Variable> for &Variable {
    type Output = Variable;
    fn add(self, other: &Variable) -> Variable {
        Add::new().unwrap().apply(&[self, other]).unwrap()
    }
}

impl std::ops::Add<&Variable> for Variable {
    type Output = Variable;
    fn add(self, other: &Variable) -> Variable {
        Add::new().unwrap().apply(&[&self, other]).unwrap()
    }
}

impl std::ops::Add<Variable> for &Variable {
    type Output = Variable;
    fn add(self, other: Variable) -> Variable {
        Add::new().unwrap().apply(&[self, &other]).unwrap()
    }
}

impl std::ops::Add<Variable> for Variable {
    type Output = Variable;
    fn add(self, other: Variable) -> Variable {
        Add::new().unwrap().apply(&[&self, &other]).unwrap()
    }
}

impl std::ops::AddAssign<&Variable> for Variable {
    fn add_assign(&mut self, other: &Variable) {
        *self = Add::new().unwrap().apply(&[self, other]).unwrap();
    }
}

impl std::ops::AddAssign<Variable> for Variable {
    fn add_assign(&mut self, other: Variable) {
        *self = Add::new().unwrap().apply(&[self, &other]).unwrap();
    }
}

#[cfg(test)]
mod broadcast_tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn add_time_emb_forward() -> MlResult<()> {
        // [N=2, C=2, H=2, W=2] + [N=2, C=2, 1, 1]
        let a = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[2, 2, 2, 2])?;
        let t = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2, 1, 1])?;
        let out = Add::new()?.forward(&[&a, &t])?.remove(0);
        assert_eq!(out.shape(), &[2, 2, 2, 2]);
        // (n=0,c=0) 의 4원소 모두 +10, (n=0,c=1) +20, (n=1,c=0) +30, (n=1,c=1) +40
        assert_eq!(
            out.data(),
            &[
                11.0, 12.0, 13.0, 14.0,
                25.0, 26.0, 27.0, 28.0,
                39.0, 40.0, 41.0, 42.0,
                53.0, 54.0, 55.0, 56.0,
            ]
        );
        Ok(())
    }

    #[cfg(feature = "enableBackward")]
    #[test]
    fn add_time_emb_backward() -> MlResult<()> {
        // grad = ones([2,2,2,2]) → grad_t 는 각 (n,c) 슬라이스의 합 = 4.0 벡터 4개
        let a = Tensor::from_vec(vec![0.0; 16], &[2, 2, 2, 2])?;
        let t = Tensor::from_vec(vec![0.0; 4], &[2, 2, 1, 1])?;
        let grad = Tensor::from_vec(vec![1.0; 16], &[2, 2, 2, 2])?;
        let grads = Add::new()?.backward(&[&a, &t], &grad)?;
        assert_eq!(grads[0].shape(), &[2, 2, 2, 2]);
        assert_eq!(grads[0].data(), &vec![1.0; 16][..]);
        assert_eq!(grads[1].shape(), &[2, 2, 1, 1]);
        assert_eq!(grads[1].data(), &[4.0, 4.0, 4.0, 4.0]);
        Ok(())
    }

    #[test]
    fn add_per_channel_last_dim() -> MlResult<()> {
        // [N=1, H=2, W=2, C=3] + [C=3]  (NumPy 기본 브로드캐스트)
        let a = Tensor::from_vec(vec![0.0; 12], &[1, 2, 2, 3])?;
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let out = Add::new()?.forward(&[&a, &b])?.remove(0);
        assert_eq!(out.shape(), &[1, 2, 2, 3]);
        assert_eq!(
            out.data(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn add_incompatible_shapes_still_error() -> MlResult<()> {
        let a = Tensor::from_vec(vec![0.0; 3], &[3])?;
        let b = Tensor::from_vec(vec![0.0; 4], &[4])?;
        assert!(Add::new()?.forward(&[&a, &b]).is_err());
        Ok(())
    }
}
