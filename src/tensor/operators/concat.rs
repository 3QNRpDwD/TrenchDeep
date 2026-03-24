use super::*;

impl Function for Concat {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Concat)
    }

    /// 텐서들을 지정한 축(axis)을 따라 이어붙입니다.
    ///
    /// # Arguments
    /// * `targets` - `[tensor1, tensor2, ..., tensorN, axis_scalar]`
    ///   마지막 원소는 이어붙일 축을 나타내는 스칼라 텐서입니다.
    ///
    /// # Returns
    /// 모든 입력 텐서를 `axis` 방향으로 이어붙인 텐서 하나를 담은 Vec.
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.len() < 3 {
            return Err(MlError::StringError(
                "Concat: 최소 2개의 텐서와 axis 스칼라가 필요합니다.".into(),
            ));
        }

        let axis = targets[targets.len() - 1].data()[0] as usize;
        let tensors = &targets[..targets.len() - 1];

        let ref_shape = tensors[0].shape();
        let ndim = ref_shape.len();

        if axis >= ndim {
            return Err(MlError::StringError(format!(
                "Concat: axis {} 는 {}차원 텐서 범위를 벗어났습니다.",
                axis, ndim
            )));
        }

        for t in tensors.iter().skip(1) {
            let s = t.shape();
            if s.len() != ndim {
                return Err(MlError::StringError(
                    "Concat: 모든 텐서의 차원 수가 동일해야 합니다.".into(),
                ));
            }
            for (i, (&a, &b)) in ref_shape.iter().zip(s.iter()).enumerate() {
                if i != axis && a != b {
                    return Err(MlError::StringError(format!(
                        "Concat: dim {} 에서 shape 불일치 ({} vs {})",
                        i, a, b
                    )));
                }
            }
        }

        // 출력 shape 계산
        let mut out_shape = ref_shape.to_vec();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();

        let total_size: usize = out_shape.iter().product();
        let mut out_data = vec![0.0f32; total_size];

        let outer_size: usize = out_shape[..axis].iter().product();
        let inner_size: usize = out_shape[axis + 1..].iter().product();

        let mut axis_offset = 0usize;
        for t in tensors.iter() {
            let t_axis_dim = t.shape()[axis];
            let t_data = t.data();

            for outer in 0..outer_size {
                let in_start = outer * t_axis_dim * inner_size;
                let out_start = outer * out_shape[axis] * inner_size + axis_offset * inner_size;
                let count = t_axis_dim * inner_size;
                out_data[out_start..out_start + count]
                    .copy_from_slice(&t_data[in_start..in_start + count]);
            }
            axis_offset += t_axis_dim;
        }

        Ok(vec![GlobalTensor::from_vec(out_data, &out_shape)?])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(
        &self,
        targets: &[&dyn TensorBase],
        grad: &dyn TensorBase,
    ) -> MlResult<Vec<GlobalTensor<f32>>> {
        let axis = targets[targets.len() - 1].data()[0] as usize;
        let tensors = &targets[..targets.len() - 1];

        let grad_data = grad.data();
        let grad_shape = grad.shape();

        let outer_size: usize = grad_shape[..axis].iter().product();
        let inner_size: usize = grad_shape[axis + 1..].iter().product();
        let grad_axis_dim = grad_shape[axis];

        let mut grads = Vec::new();
        let mut axis_offset = 0usize;

        for t in tensors.iter() {
            let t_shape = t.shape();
            let t_axis_dim = t_shape[axis];
            let t_total: usize = t_shape.iter().product();
            let mut t_grad = vec![0.0f32; t_total];

            for outer in 0..outer_size {
                let grad_start =
                    outer * grad_axis_dim * inner_size + axis_offset * inner_size;
                let t_start = outer * t_axis_dim * inner_size;
                let count = t_axis_dim * inner_size;
                t_grad[t_start..t_start + count]
                    .copy_from_slice(&grad_data[grad_start..grad_start + count]);
            }

            grads.push(GlobalTensor::from_vec(t_grad, t_shape)?);
            axis_offset += t_axis_dim;
        }

        // axis 스칼라는 기울기 없음
        grads.push(GlobalTensor::from_vec(vec![0.0], &[1, 1])?);

        Ok(grads)
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_last_dim() -> MlResult<()> {
        // [1, 4] + [1, 4] → axis=1 → [1, 8]
        let a = GlobalTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4])?;
        let b = GlobalTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[1, 4])?;
        let axis = GlobalTensor::from_vec(vec![1.0], &[1, 1])?;

        let result = Concat::new()?.forward(&[&a, &b, &axis])?.remove(0);

        assert_eq!(result.shape(), &[1, 8]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        Ok(())
    }

    #[test]
    fn concat_3d_last_dim() -> MlResult<()> {
        // [2, 3, 4] + [2, 3, 4] → axis=2 → [2, 3, 8]
        let n = 2 * 3 * 4;
        let a = GlobalTensor::from_vec((0..n).map(|x| x as f32).collect(), &[2, 3, 4])?;
        let b = GlobalTensor::from_vec((0..n).map(|x| x as f32 + 100.0).collect(), &[2, 3, 4])?;
        let axis = GlobalTensor::from_vec(vec![2.0], &[1, 1])?;

        let result = Concat::new()?.forward(&[&a, &b, &axis])?.remove(0);

        assert_eq!(result.shape(), &[2, 3, 8]);
        Ok(())
    }
}
