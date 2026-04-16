use crate::tensor::operators::ReshapeOp;
use super::*;

impl Reshape {
    /// `target_shape`에 `-1`을 하나만 사용하면 해당 차원을 자동 추론합니다.
    ///
    /// # 예시
    /// ```ignore
    /// // [N, C, H, W] → [N, H*W, C]  (batch 차원 유지)
    /// Reshape::new(&[0, -1, 0], "flatten_spatial")?;
    /// // 0 = 입력의 해당 차원을 그대로 사용 (런타임에 결정)
    ///
    /// // 고정 shape
    /// Reshape::new(&[4, 256], "fixed_reshape")?;
    /// ```
    pub fn new(target_shape: &[isize], label: &str) -> MlResult<Self> {
        let neg_count = target_shape.iter().filter(|&&d| d < 0).count();
        if neg_count > 1 {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: vec![],
                got: target_shape.iter().map(|&d| d as usize).collect(),
            }));
        }
        Ok(Self {
            label: label.to_string(),
            target_shape: target_shape.to_vec(),
            operator: ReshapeOp::new()?,
        })
    }

    /// 입력 shape를 기반으로 실제 target shape를 계산합니다.
    /// `-1`은 나머지 원소 수로 추론하고, `0`은 입력의 해당 차원을 그대로 복사합니다.
    fn resolve_shape(&self, input_shape: &[usize]) -> MlResult<Vec<usize>> {
        let total: usize = input_shape.iter().product();
        let mut resolved: Vec<usize> = Vec::with_capacity(self.target_shape.len());
        let mut infer_idx: Option<usize> = None;
        let mut known_product: usize = 1;

        for (i, &dim) in self.target_shape.iter().enumerate() {
            if dim == 0 {
                // 입력의 해당 차원을 그대로 사용
                if i >= input_shape.len() {
                    return Err(MlError::TensorError(TensorError::InvalidShape {
                        expected: self.target_shape.iter().map(|&d| d as usize).collect(),
                        got: input_shape.to_vec(),
                    }));
                }
                resolved.push(input_shape[i]);
                known_product *= input_shape[i];
            } else if dim < 0 {
                // 추론 대상
                infer_idx = Some(i);
                resolved.push(0); // placeholder
            } else {
                resolved.push(dim as usize);
                known_product *= dim as usize;
            }
        }

        if let Some(idx) = infer_idx {
            if known_product == 0 || total % known_product != 0 {
                return Err(MlError::TensorError(TensorError::InvalidShape {
                    expected: self.target_shape.iter().map(|&d| d as usize).collect(),
                    got: input_shape.to_vec(),
                }));
            }
            resolved[idx] = total / known_product;
        }

        Ok(resolved)
    }
}

impl Layer for Reshape {
    #[cfg(all(feature = "enableBackward"))]
    fn apply(&mut self, input: &Variable) -> MlResult<Variable> {
        let resolved = self.resolve_shape(input.tensor().shape())?;
        let size: usize = resolved.iter().product();
        // Reshape 연산자는 targets[1].shape()를 새 shape로 사용
        let shape_dummy = Variable::new(Tensor::from_vec(vec![0.0f32; size], &resolved)?);
        self.operator.apply(&[input, &shape_dummy])
    }

    fn predict(&self, input: &dyn TensorBase) -> MlResult<GlobalTensor<f32>> {
        let resolved = self.resolve_shape(input.shape())?;
        Ok(GlobalTensor::from_vec(input.data().to_vec(), &resolved)?)
    }

    fn params(&self) -> Vec<&dyn Parameter> {
        vec![]
    }

    fn label(&self) -> &str {
        &self.label
    }
}
