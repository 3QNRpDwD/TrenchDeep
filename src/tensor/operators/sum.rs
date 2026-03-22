use super::*;

impl Function for Sum {
    fn new() -> MlResult<GlobalFunction> {
        register_operator!(Sum)
    }

    fn forward(&self, inputs: &[&dyn TensorBase]) -> MlResult<Vec<GlobalTensor<f32>>> {
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
        Ok(vec![GlobalTensor::from_vec(vec![total_sum], &[1,1])?])
    }

    #[cfg(feature = "enableBackward")]
    fn backward(&self, targets: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<GlobalTensor<f32>>> {
        if targets.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let gt = GlobalTensor { data: grad.data().to_vec(), shape: grad.shape().to_vec(), dirty: false };
        Ok(vec![gt.clone(); targets.len()])
    }
    
    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &NodeId { &self.node_id }
}