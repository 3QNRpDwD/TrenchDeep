use super::*;

impl Function<f32> for Mul {
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
    fn forward(&self, targets: &[&Tensor<f32>]) -> MlResult<Vec<Tensor<f32>>> {
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

            Ok(vec![Tensor::<f32>::from_vec(result, shape1)?])
        } else {
            // 기존 코드 유지
            match targets[0].chk_shape(targets[1]) {
                Err(e) => Err(e),
                _ => Ok(vec![Tensor::<f32>::from_vec(
                    self.backend().multiply(targets[0].data(), targets[1].data()),
                    shape1
                )?])
            }
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, targets: &[&Tensor<f32>], grad: &Tensor<f32>) -> MlResult<Vec<Tensor<f32>>> {
        Ok(vec![
            self.forward(&[grad, targets[1]])?.remove(0),
            self.forward(&[grad, targets[0]])?.remove(0)
        ])
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
impl std::ops::Mul<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: Tensor<f32>) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[&self, &other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: &Tensor<f32>) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[&self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<&Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: &Tensor<f32>) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, other]).unwrap().remove(0))
    }
}

impl std::ops::Mul<Tensor<f32>> for &Tensor<f32> {
    type Output = Tensor<f32>;

    fn mul(self, other: Tensor<f32>) -> Self::Output {
        OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, &other]).unwrap().remove(0))
    }
}

/// MulAssign trait implementation for Tensor
impl std::ops::MulAssign<Tensor<f32>> for Tensor<f32> {
    fn mul_assign(&mut self, other: Tensor<f32>) {
        *self =  OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, &other]).unwrap().remove(0));
    }
}

impl std::ops::MulAssign<&Tensor<f32>> for Tensor<f32> {
    fn mul_assign(&mut self, other: &Tensor<f32>) {
        *self =  OPERATOR_STORAGE.with(|ops| ops.borrow().get("Mul").unwrap().forward(&[self, other]).unwrap().remove(0));
    }
}
