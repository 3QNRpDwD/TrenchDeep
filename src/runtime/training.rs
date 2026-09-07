use super::*;
pub(crate) struct TrainingScope {
    context: ExecutionContext,
    finished: bool,
}
impl ExecutionContext {
    /// Execute one training unit with exclusive graph ownership.
    ///
    /// Rejects nested training and pre-existing graphs before invoking `operation`.
    /// Graphs and gradients are cleaned on success and on returned errors. If both
    /// the operation and cleanup fail, both causes are returned in `CleanupError`.
    /// Parameter updates are not rolled back. Panic unwinding performs best-effort
    /// cleanup through the private scope guard.
    pub fn with_training_scope<T>(&self, operation: impl FnOnce() -> MlResult<T>) -> MlResult<T> {
        let scope = self.begin_training_scope()?;
        scope.finish(operation())
    }

    pub(crate) fn begin_training_scope(&self) -> MlResult<TrainingScope> {
        self.collect()?;
        let mut state = self
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if self.inner.training_active.get() || !state.engine()?.nodes().is_empty() {
            return Err(ContextError::ActiveGraphConflict.into());
        }
        state.gradients.clear();
        self.inner.training_active.set(true);
        Ok(TrainingScope {
            context: self.clone(),
            finished: false,
        })
    }
}
impl TrainingScope {
    fn cleanup(&mut self) -> MlResult<()> {
        let mut state = self
            .context
            .inner
            .state
            .try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        state.clear_graph()?;
        state.gradients.clear();
        state.collect()?;
        self.context.inner.training_active.set(false);
        self.finished = true;
        Ok(())
    }
    pub(crate) fn finish<T>(mut self, result: MlResult<T>) -> MlResult<T> {
        match (result, self.cleanup()) {
            (Ok(v), Ok(())) => Ok(v),
            (Err(e), Ok(())) | (Ok(_), Err(e)) => Err(e),
            (Err(primary), Err(cleanup)) => Err(MlError::CleanupError {
                primary: Box::new(primary),
                cleanup: Box::new(cleanup),
            }),
        }
    }
}
impl Drop for TrainingScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.cleanup();
        }
    }
}
