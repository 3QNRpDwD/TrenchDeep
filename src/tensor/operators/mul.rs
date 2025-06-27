use super::*;

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
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let shape1 = targets[0].shape();
        let shape2 = targets[1].shape();

        // [1,1] 텐서인 경우에만 브로드캐스팅
        if shape2 == &[1, 1] {
            let target2_data = targets[1].data();
            let scalar_value = target2_data[0];
            // 첫 번째 텐서의 모든 원소에 스칼라 값을 곱함
            let result = targets[0].data()
                .iter()
                .map(|&x| x * scalar_value)
                .collect::<Vec<f32>>();

            Ok(vec![PooledTensor::from_vec(result, shape1)?])
        } else {
            // 기존 코드 유지
            match targets[0].chk_shape(targets[1]) {
                Err(e) => Err(e),
                _ => Ok(vec![PooledTensor::from_vec(
                    self.backend().multiply(targets[0].data(), targets[1].data()),
                    shape1
                )?])
            }
        }
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: HandleId) -> MlResult<Vec<Tensor>> {
        let shape1 = targets[0].shape();
        let shape2 = targets[1].shape();

        // [1,1] 텐서인 경우에만 브로드캐스팅
        if shape2 == &[1, 1] {
            let target2_data = targets[1].data();
            let scalar_value = target2_data[0];
            // 첫 번째 텐서의 모든 원소에 스칼라 값을 곱함
            let result = targets[0].data()
                .iter()
                .map(|&x| x * scalar_value)
                .collect::<Vec<f32>>();

            Ok(vec![Tensor::with_id(result, shape1, node_id)?])
        } else {
            // 기존 코드 유지
            match targets[0].chk_shape(targets[1]) {
                Err(e) => Err(e),
                _ => Ok(vec![Tensor::with_id(
                    self.backend().multiply(targets[0].data(), targets[1].data()),
                    shape1,
                    node_id
                )?])
            }
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        Ok(vec![
            self.forward(&[grad, targets[1]])?.remove(0),
            self.forward(&[grad, targets[0]])?.remove(0)
        ])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
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
    type Output = PooledTensor;

    fn mul(self, other: Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().forward(&[&self, &other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = PooledTensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().forward(&[&self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = PooledTensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&dyn TensorBase> for &dyn TensorBase {
    type Output = PooledTensor;

    fn mul(self, other: &dyn TensorBase) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = PooledTensor;

    fn mul(self, other: Tensor) -> Self::Output {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().forward(&[self, &other]).unwrap().remove(0))
    }
}

/// MulAssign trait implementation for Tensor
impl std::ops::MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, other: Tensor) {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().assign_forward(&[self, &other], self.0).unwrap().remove(0));
    }
}

impl std::ops::MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        Mul::new().unwrap();
        OPERATOR_STORAGE.with(|ops| ops.borrow_mut().get_mut("Mul").unwrap().assign_forward(&[self, other], self.0).unwrap().remove(0));
    }
}
