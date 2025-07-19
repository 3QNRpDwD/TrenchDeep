use super::*;

/// Transpose 연산의 핵심 로직을 수행하는 헬퍼 함수
fn transpose_core(input: &dyn TensorBase, permutation: &[usize]) -> MlResult<PooledTensor> {
    let rank = input.shape().len();
    if rank != permutation.len() {
        return Err(MlError::StringError(
            format!(
                "Permutation length ({}) does not match tensor rank ({}).",
                permutation.len(),
                rank
            ),
        ));
    }

    // 새 shape 및 stride 계산
    let old_shape = input.shape();
    let new_shape: Vec<usize> = permutation.iter().map(|&p| old_shape[p]).collect();

    let mut old_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }

    let mut new_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    // 데이터 전치
    let mut result_data = vec![0.0; input.data().len()];
    let mut source_coords = vec![0usize; rank];
    for i in 0..input.data().len() {
        // 원본 텐서의 1차원 인덱스(i)로부터 다차원 좌표(source_coords) 계산
        let mut temp_idx = i;
        for j in 0..rank {
            source_coords[j] = temp_idx / old_strides[j];
            temp_idx %= old_strides[j];
        }

        // 원본 좌표를 permutation에 따라 변환하여 목적지 좌표(target_coords) 계산
        let mut target_coords = vec![0; rank];
        for j in 0..rank {
            target_coords[permutation[j]] = source_coords[j];
        }

        // 목적지 좌표로부터 1차원 인덱스(target_idx) 계산
        let mut target_idx = 0;
        for j in 0..rank {
            target_idx += target_coords[j] * new_strides[j];
        }

        result_data[target_idx] = input.data()[i];
    }

    Ok(PooledTensor::from_vec(result_data, &new_shape)?)
}

impl Function for Transpose {
    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let input_tensor = inputs[0];
        let dims_tensor = inputs[1];

        let permutation: Vec<usize> = dims_tensor.data().iter().map(|&x| x as usize).collect();

        transpose_core(input_tensor, &permutation)
            .map(|tensor| vec![tensor])
    }

    #[cfg(feature = "enableBackpropagation")]
    fn backward(&self, inputs: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        if inputs.len() != 2 {
            return Err(MlError::TensorError(TensorError::InvalidInputCount { expected: 2, got: inputs.len() }));
        }
        let dims_tensor = inputs[1];
        let permutation: Vec<usize> = dims_tensor.data().iter().map(|&x| x as usize).collect();
        let rank = permutation.len();

        // 역전파를 위해 역순(inverse) permutation을 계산
        let mut inv_permutation = vec![0; rank];
        for i in 0..rank {
            inv_permutation[permutation[i]] = i;
        }

        transpose_core(grad, &inv_permutation)
            .map(|tensor| vec![tensor])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

#[cfg(test)]
mod tests {
    use crate::nn::{Parameter, Variable};
    use crate::tensor::operators::tests::assert_tensor_eq;
    use crate::tensor::operators::Transpose;
    use crate::tensor::AutogradFunction;
    use crate::{tensor::{Tensor, TensorBase}, variable, MlResult};

    #[test]
    fn test_transpose_backward() -> MlResult<()> {
        let a = variable!(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // 연산자 생성 시 더 이상 dims를 넘기지 않음
        let op = Transpose::new();
        // 차원 순서 정보를 담는 텐서를 별도로 생성 (0, 1) -> (1, 0)
        let dims_tensor = Variable::new(Tensor::from_vec(vec![1.0, 0.0], &[2])?);

        // apply 함수에 두 텐서를 전달
        let output = op.apply(&[&a, &dims_tensor])?;

        output.backward()?;

        let grad_a = a.grad();
        // 그래디언트는 원래 텐서와 동일한 shape를 가져야 함
        let expected_grad = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2])?;
        assert_tensor_eq(grad_a, &expected_grad)?;

        Ok(())
    }
}