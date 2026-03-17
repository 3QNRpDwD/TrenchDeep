use std::collections::HashMap;
use super::*;

// 역전파 연산 클로저의 타입 별칭입니다.
// 이제 Gradients 맵과 함께 TensorAllocator에 대한 가변 참조도 받습니다.
type BackwardOp = Box<dyn FnOnce(&mut Gradients, &mut TensorAllocator) -> MlResult<()>>;

/// GradientTape는 역전파 시 실행될 연산의 시퀀스를 저장합니다.
/// 각 Layer는 자신만의 GradientTape 인스턴스를 소유하게 됩니다.
#[derive(Default)]
pub struct GradientTape {
    ops: Vec<BackwardOp>,
}

/// Gradients는 계산된 기울기 텐서들의 HandleId를 저장하는 컨테이너입니다.
/// Key: 변수의 HandleId, Value: 해당 변수의 기울기 텐서의 HandleId
#[derive(Debug, Default)]
pub struct Gradients {
    grads: HashMap<HandleId, HandleId>,
}

impl GradientTape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_op(&mut self, op: BackwardOp) {
        self.ops.push(op);
    }

    pub fn clear(&mut self) {
        self.ops.clear();
    }

    /// 역전파를 실행합니다.
    /// 초기 기울기 값을 포함하는 Gradients 맵과 기울기 전용 Allocator를 받아,
    /// 테이프에 기록된 모든 연산을 역순으로 실행하며 Gradients 맵을 채웁니다.
    pub fn backward(&mut self, mut grads: Gradients, alloc: &mut TensorAllocator) -> MlResult<Gradients> {
        while let Some(op) = self.ops.pop() {
            op(&mut grads, alloc)?;
        }
        Ok(grads)
    }
}

impl Gradients {
    pub fn new() -> Self {
        Self::default()
    }

    /// 특정 HandleId에 해당하는 기울기 텐서의 HandleId를 반환합니다.
    pub fn get(&self, id: &HandleId) -> Option<&HandleId> {
        self.grads.get(id)
    }

    /// 특정 HandleId에 새로운 기울기 HandleId를 설정합니다.
    pub fn set(&mut self, id: HandleId, grad_id: HandleId) {
        self.grads.insert(id, grad_id);
    }

    /// 특정 HandleId에 기울기를 누적합니다.
    pub fn accumulate(&mut self, id: HandleId, grad_to_add_id: HandleId, alloc: &mut TensorAllocator) -> MlResult<()> {
        todo!()
    }
}
