use super::*;

impl Function for Mul {
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

            Ok(vec![PooledTensor::from_vec(result, shape1).unwrap()])
        } else {
            // 기존 코드 유지
            match targets[0].chk_shape(targets[1]) {
                Err(e) => Err(e),
                _ => Ok(vec![PooledTensor::from_vec(
                    self.backend().multiply(targets[0].data(), targets[1].data()),
                    shape1
                ).unwrap()])
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

            Ok(vec![Tensor::with_id(result, shape1, node_id).unwrap()])
        } else {
            // 기존 코드 유지
            match targets[0].chk_shape(targets[1]) {
                Err(e) => Err(e),
                _ => Ok(vec![Tensor::with_id(
                    self.backend().multiply(targets[0].data(), targets[1].data()),
                    shape1,
                    node_id
                ).unwrap()])
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

impl std::ops::Mul<Tensor> for Tensor {
    type Output = PooledTensor;

    fn mul(self, other: Tensor) -> Self::Output {
        let op = Mul::new();
        op.forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = PooledTensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        let op = Mul::new();
        op.forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = PooledTensor;

    fn mul(self, other: &Tensor) -> Self::Output {
        let op = Mul::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<&dyn TensorBase> for &dyn TensorBase {
    type Output = PooledTensor;

    fn mul(self, other: &dyn TensorBase) -> Self::Output {
        let op = Mul::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = PooledTensor;

    fn mul(self, other: Tensor) -> Self::Output {
        let op = Mul::new();
        op.forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, other: Tensor) {
        let op = Mul::new();
        op.assign_forward(&[self, &other], self.0).unwrap();
    }
}

impl std::ops::MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, other: &Tensor) {
        let op = Mul::new();
        op.assign_forward(&[self, other], self.0).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MlResult, variable,tensor::{TensorBase, Tensor}};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn test_mul_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = variable!(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let op = Mul::new();
        let output = op.apply(&[&a, &b])?;

        output.backward()?;

        let grad_a = a.grad();
        let grad_b = b.grad();

        let expected_grad_a = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let expected_grad_b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad_a)?;
        assert_tensor_eq(grad_b, &expected_grad_b)?;

        Ok(())
    }
}