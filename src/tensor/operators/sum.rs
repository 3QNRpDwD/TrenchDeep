use super::*;

impl Function for Sum {
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        // Sum 함수는 하나의 입력 텐서만 받습니다.
        if inputs.len() != 1 {
            return Err(MlError::StringError(format!(
                "Sum operation expects 1 input tensor, but got {}",
                inputs.len()
            )));
        }
        let target = inputs[0];

        // 텐서 데이터의 모든 요소의 합을 계산합니다.
        // f32 타입을 가정합니다.
        let total_sum: f32 = target.data().iter().sum();

        // 결과를 shape이 [1]인 새로운 텐서(스칼라)로 만들어 반환합니다.
        Ok(vec![PooledTensor::from_vec(vec![total_sum], &[1,1]).unwrap()])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        if targets.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let gt = PooledTensor::from_vec(grad.data().to_vec(), grad.shape()).unwrap();
        Ok(vec![gt.clone(); targets.len()])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MlResult, variable,tensor::{TensorBase, Tensor}};
    use crate::tensor::operators::tests::assert_tensor_eq;

    #[test]
    fn test_sum_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let op = Sum::new();
        let output = op.apply(&[&a])?;

        output.backward()?;

        let grad_a = a.grad();
        let expected_grad = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}