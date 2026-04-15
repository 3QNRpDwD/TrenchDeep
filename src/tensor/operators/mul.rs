use super::*;
use crate::nn::Variable;
use crate::tensor::AutogradFunction;
use crate::tensor::broadcast::{broadcast_offsets, broadcast_shape, reduce_to_shape};

impl Function for Mul {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Mul)
    }
    
    /// Multiplies two tensors element-wise
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply the current tensor by
    ///
    /// # Returns
    /// A new tensor with the result of the element-wise multiplication
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        let shape1 = targets[0].shape();
        let shape2 = targets[1].shape();

        #[cfg(feature = "debugging")]
        tracing::debug!(
            "[Mul::forward] {} ⊙ {}",
            crate::tensor::operators::debug::summary("lhs", targets[0]),
            crate::tensor::operators::debug::summary("rhs", targets[1])
        );

        // [1,1] 텐서인 경우에만 브로드캐스팅 (hot path)
        if shape2 == &[1, 1] {
            let scalar_value = targets[1].data()[0];
            let result = targets[0].data()
                .iter()
                .map(|&x| x * scalar_value)
                .collect::<Vec<f32>>();
            return Ok(vec![GlobalTensor::from_vec(result, shape1)?]);
        }

        if shape1 == shape2 {
            return Ok(vec![GlobalTensor::from_vec(
                self.backend().multiply(targets[0].data(), targets[1].data()),
                shape1,
            )?]);
        }

        // 범용 브로드캐스트 경로
        let out_shape = broadcast_shape(shape1, shape2)?;
        let a = targets[0].data();
        let b = targets[1].data();
        let offsets = broadcast_offsets(shape1, shape2, &out_shape);
        let data: Vec<f32> = offsets.iter().map(|&(ao, bo)| a[ao] * b[bo]).collect();
        Ok(vec![GlobalTensor::from_vec(data, &out_shape)?])
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: NodeId) -> MlResult<Vec<Tensor>> {
        let shape1 = targets[0].shape();
        let shape2 = targets[1].shape();

        if shape2 == &[1, 1] {
            let scalar_value = targets[1].data()[0];
            let result = targets[0].data()
                .iter()
                .map(|&x| x * scalar_value)
                .collect::<Vec<f32>>();
            return Ok(vec![Tensor::with_id(result, shape1, node_id)?]);
        }

        if shape1 == shape2 {
            return Ok(vec![Tensor::with_id(
                self.backend().multiply(targets[0].data(), targets[1].data()),
                shape1,
                node_id,
            )?]);
        }

        let out_shape = broadcast_shape(shape1, shape2)?;
        let a = targets[0].data();
        let b = targets[1].data();
        let offsets = broadcast_offsets(shape1, shape2, &out_shape);
        let data: Vec<f32> = offsets.iter().map(|&(ao, bo)| a[ao] * b[bo]).collect();
        Ok(vec![Tensor::with_id(data, &out_shape, node_id)?])
    }

    #[cfg(all(feature = "enableBackward"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        let shape_a = targets[0].shape();
        let shape_b = targets[1].shape();

        let full_grad_a = self.forward(&[grad, targets[1]])?.remove(0);
        let full_grad_b = self.forward(&[grad, targets[0]])?.remove(0);

        let grad_a = if full_grad_a.shape == shape_a {
            full_grad_a
        } else {
            let reduced = reduce_to_shape(&full_grad_a.data, &full_grad_a.shape, shape_a);
            GlobalTensor::from_vec(reduced, shape_a)?
        };
        let grad_b = if full_grad_b.shape == shape_b {
            full_grad_b
        } else {
            let reduced = reduce_to_shape(&full_grad_b.data, &full_grad_b.shape, shape_b);
            GlobalTensor::from_vec(reduced, shape_b)?
        };

        let result = vec![grad_a, grad_b];

        #[cfg(feature = "debugging")]
        {
            crate::tensor::operators::debug::stats_raw("  └─ dlhs", &result[0].data, &result[0].shape);
            crate::tensor::operators::debug::stats_raw("  └─ drhs", &result[1].data, &result[1].shape);
        }

        Ok(result)
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}


/// Multiply trait implementation for owned tensors
///
/// # Arguments
/// * `other` - The tensor to multiply with self
///
/// # Returns
/// A new tensor containing the element-wise product (Hadamard product)
///
/// # Note
/// * This performs element-wise multiplication, not matrix multiplication
/// * For matrix multiplication, use `matmul()` instead
impl std::ops::Mul<Tensor> for Tensor {
    type Output = GlobalTensor<f32>;

    fn mul(self, other: Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[&self, &other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = GlobalTensor<f32>;

    fn mul(self, other: &Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[&self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = GlobalTensor<f32>;

    fn mul(self, other: &Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&dyn TensorBase> for &dyn TensorBase {
    type Output = GlobalTensor<f32>;

    fn mul(self, other: &dyn TensorBase) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = GlobalTensor<f32>;

    fn mul(self, other: Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, &other]).unwrap().remove(0))
    }
}

/// MulAssign trait implementation for Tensor
impl std::ops::MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, other: Tensor) {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().assign_forward(&[self, &other], self.id()).unwrap().remove(0));
    }
}

impl std::ops::MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().assign_forward(&[self, other], self.id()).unwrap().remove(0));
    }
}

// Variable operator overloading (graph-tracked)
impl std::ops::Mul<&Variable> for &Variable {
    type Output = Variable;
    fn mul(self, other: &Variable) -> Variable {
        Mul::new().unwrap().apply(&[self, other]).unwrap()
    }
}

impl std::ops::Mul<&Variable> for Variable {
    type Output = Variable;
    fn mul(self, other: &Variable) -> Variable {
        Mul::new().unwrap().apply(&[&self, other]).unwrap()
    }
}

impl std::ops::Mul<Variable> for &Variable {
    type Output = Variable;
    fn mul(self, other: Variable) -> Variable {
        Mul::new().unwrap().apply(&[self, &other]).unwrap()
    }
}

impl std::ops::Mul<Variable> for Variable {
    type Output = Variable;
    fn mul(self, other: Variable) -> Variable {
        Mul::new().unwrap().apply(&[&self, &other]).unwrap()
    }
}

#[cfg(test)]
mod broadcast_tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn mul_scalar_hot_path() -> MlResult<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let s = Tensor::from_vec(vec![3.0], &[1, 1])?;
        let out = Mul::new()?.forward(&[&a, &s])?.remove(0);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[3.0, 6.0, 9.0, 12.0]);
        Ok(())
    }

    #[test]
    fn mul_time_emb_forward() -> MlResult<()> {
        // [N=2,C=2,H=2,W=2] * [N=2,C=2,1,1]
        let a = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[2, 2, 2, 2])?;
        let s = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1, 1])?;
        let out = Mul::new()?.forward(&[&a, &s])?.remove(0);
        assert_eq!(out.shape(), &[2, 2, 2, 2]);
        assert_eq!(
            out.data(),
            &[
                1.0, 2.0, 3.0, 4.0,
                10.0, 12.0, 14.0, 16.0,
                27.0, 30.0, 33.0, 36.0,
                52.0, 56.0, 60.0, 64.0,
            ]
        );
        Ok(())
    }

    #[test]
    fn mul_mask_forward() -> MlResult<()> {
        // [N=2, L=2, L=2] * [1, L=2, L=2]
        let a = Tensor::from_vec((1..=8).map(|x| x as f32).collect(), &[2, 2, 2])?;
        let m = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 2, 2])?;
        let out = Mul::new()?.forward(&[&a, &m])?.remove(0);
        assert_eq!(out.shape(), &[2, 2, 2]);
        assert_eq!(out.data(), &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0]);
        Ok(())
    }

    #[cfg(feature = "enableBackward")]
    #[test]
    fn mul_time_emb_backward() -> MlResult<()> {
        // a=[2,2,2,2], s=[2,2,1,1], grad=ones → da=s(broadcast), ds=sum over H,W of a
        let a = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[2, 2, 2, 2])?;
        let s = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2, 1, 1])?;
        let grad = Tensor::from_vec(vec![1.0; 16], &[2, 2, 2, 2])?;
        let grads = Mul::new()?.backward(&[&a, &s], &grad)?;
        assert_eq!(grads[0].shape(), &[2, 2, 2, 2]);
        // da[n,c,h,w] = s[n,c,0,0]
        assert_eq!(
            grads[0].data(),
            &[
                10.0, 10.0, 10.0, 10.0,
                20.0, 20.0, 20.0, 20.0,
                30.0, 30.0, 30.0, 30.0,
                40.0, 40.0, 40.0, 40.0,
            ]
        );
        assert_eq!(grads[1].shape(), &[2, 2, 1, 1]);
        // ds[n,c] = sum_{h,w} a[n,c,h,w]
        // (0,0): 1+2+3+4=10, (0,1): 5+6+7+8=26, (1,0): 9+10+11+12=42, (1,1): 13+14+15+16=58
        assert_eq!(grads[1].data(), &[10.0, 26.0, 42.0, 58.0]);
        Ok(())
    }
}
