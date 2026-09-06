use super::*;

pub(crate) struct TrainingScope {
    context: ExecutionContext,
    finished: bool,
}

impl ExecutionContext {
    pub(crate) fn begin_training_scope(&self) -> MlResult<TrainingScope> {
        let mut state = self.runtime.state.try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        if self.runtime.training_active.get() || !state.graph.is_empty() {
            return Err(ContextError::ActiveGraphConflict.into());
        }
        state.gradients.clear();
        self.runtime.training_active.set(true);
        Ok(TrainingScope { context: self.clone(), finished: false })
    }
}

impl TrainingScope {
    fn cleanup(&mut self) -> MlResult<()> {
        let mut state = self.context.runtime.state.try_borrow_mut()
            .map_err(|_| ContextError::BorrowConflict)?;
        let outputs = state.graph.keys().copied().collect::<Vec<_>>();
        for output in outputs { state.remove_graph_node(output); }
        state.gradients.clear();
        state.collect_garbage();
        self.context.runtime.training_active.set(false);
        self.finished = true;
        Ok(())
    }

    pub(crate) fn finish<T>(mut self, result: MlResult<T>) -> MlResult<T> {
        match (result, self.cleanup()) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(primary), Err(cleanup)) => Err(crate::MlError::CleanupError {
                primary: Box::new(primary), cleanup: Box::new(cleanup),
            }),
        }
    }
}

impl Drop for TrainingScope {
    fn drop(&mut self) {
        if !self.finished {
            // During unwind the enclosing kernel borrow has already been released.
            // If an external borrow still prevents cleanup, keep the scope marked
            // active so a subsequent training run cannot silently reuse dirty state.
            let _ = self.cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_rejects_existing_graph_and_nested_execution() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let y = x.square()?;
        assert!(matches!(ctx.begin_training_scope(), Err(crate::MlError::ContextError(ContextError::ActiveGraphConflict))));
        y.backward()?;
        let scope = ctx.begin_training_scope()?;
        assert!(x.grad()?.is_none());
        assert!(ctx.begin_training_scope().is_err());
        x.square()?.backward()?;
        scope.finish(Ok(()))?;
        assert!(x.grad()?.is_none());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        Ok(())
    }
    #[test]
    fn scope_cleans_graph_during_unwind() -> MlResult<()> {
        let ctx = ExecutionContext::new();
        let x = ctx.parameter(vec![2.0], &[])?;
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = ctx.begin_training_scope().unwrap();
            let _output = x.square().unwrap();
            panic!("model failed");
        }));
        assert!(panic.is_err());
        assert_eq!(ctx.graph_stats()?.graph_nodes, 0);
        ctx.begin_training_scope()?.finish(Ok(()))
    }
}
