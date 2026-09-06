use crate::contracts::*;
use crate::{ContextError, MlResult};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, Default)]
pub struct CpuTensorStore { buffers: HashMap<TensorId, Rc<RefCell<TensorBuffer>>> }
impl TensorStore for CpuTensorStore {
    fn insert(&mut self, id: TensorId, buffer: TensorBuffer) -> MlResult<()> {
        if self.buffers.contains_key(&id) {
            return Err(crate::TensorError::InvalidOperation { op: "insert", reason: "duplicate tensor ID".into() }.into());
        }
        self.buffers.insert(id, Rc::new(RefCell::new(buffer))); Ok(())
    }
    fn alias(&mut self, id: TensorId, source: TensorId) -> MlResult<()> {
        if self.buffers.contains_key(&id) {
            return Err(crate::TensorError::InvalidOperation { op: "alias", reason: "duplicate tensor ID".into() }.into());
        }
        let buffer = self.buffers.get(&source).ok_or(ContextError::UnknownTensor(source))?.clone();
        self.buffers.insert(id, buffer); Ok(())
    }
    fn with_view(&self, id: TensorId, visitor: &mut dyn FnMut(TensorView<'_>) -> MlResult<()>) -> MlResult<()> {
        let buffer = self.buffers.get(&id).ok_or(ContextError::UnknownTensor(id))?;
        let buffer = buffer.try_borrow().map_err(|_| ContextError::BorrowConflict)?;
        visitor(buffer.view())
    }
    fn replace(&mut self, id: TensorId, buffer: TensorBuffer) -> MlResult<()> {
        let old = self.buffers.get(&id).ok_or(ContextError::UnknownTensor(id))?;
        let mut old = old.try_borrow_mut().map_err(|_| ContextError::BorrowConflict)?;
        if old.shape() != buffer.shape() {
            return Err(crate::TensorError::InvalidShape { expected: old.shape().to_vec(), got: buffer.shape().to_vec() }.into());
        }
        *old = buffer; Ok(())
    }
    fn remove(&mut self, id: TensorId) -> MlResult<()> { self.buffers.remove(&id); Ok(()) }
}
