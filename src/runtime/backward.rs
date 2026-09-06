use super::*;
impl ExecutionContext {
    pub fn backward(&self,output:&Variable,options:BackwardOptions<'_>)->MlResult<()> {
        self.validate(output.tensor())?;
        let mut state=self.inner.state.try_borrow_mut().map_err(|_|ContextError::BorrowConflict)?;
        state.engine()?;
        let root=output.tensor.id();
        if !state.tracked.contains(&root){return Err(AutogradError::NodeNotFound(root).into());}
        if state.consumed.contains(&root){return Err(AutogradError::GraphAlreadyFreed(root).into());}
        let value=state.snapshot(root)?;
        let seed=if let Some(gradient)=options.gradient {
            if gradient.context_id()!=self.id {return Err(ContextError::Mismatch.into());}
            let grad=state.snapshot(gradient.id())?;
            if grad.shape!=value.shape {return Err(AutogradError::GradientShapeMismatch {expected:value.shape,got:grad.shape}.into());}grad
        }else{
            if value.data.len()!=1 {return Err(AutogradError::OutputNotScalar(value.shape).into());}
            TensorBuffer::from_vec(vec![1.0],&value.shape)?
        };
        let order=state.engine()?.order(root)?;
        let mut seen=HashSet::from([root]);
        let mut records=Vec::new();
        for &id in &order {
            if !seen.insert(id) && records.iter().any(|n:&GradientRecord|n.output==id) {return Err(AutogradError::CycleDetected.into());}
            let record=state.engine()?.get(id).ok_or(AutogradError::NodeNotFound(id))?;
            seen.extend(record.inputs.iter().copied());records.push(record);
        }
        state.gradients.retain(|id,_|!seen.contains(id));
        state.gradients.insert(root,seed);
        for record in records.iter().rev() {
            let Some(grad)=state.gradients.get(&record.output).cloned() else{continue;};
            let inputs=record.inputs.iter().map(|&id|state.snapshot(id)).collect::<MlResult<Vec<_>>>()?;
            let saved=record.saved.iter().map(|&id|state.snapshot(id)).collect::<MlResult<Vec<_>>>()?;
            let views=inputs.iter().map(TensorBuffer::view).collect::<Vec<_>>();
            let saved_views=saved.iter().map(TensorBuffer::view).collect::<Vec<_>>();
            let incoming=record.backward.backward(&views,&saved_views,grad.view())?;
            if incoming.len()!=record.inputs.len(){return Err(AutogradError::BackwardArityMismatch {expected:record.inputs.len(),got:incoming.len()}.into());}
            // Validate the complete VJP before applying any of its gradients.
            for (input,g) in inputs.iter().zip(&incoming) {if let Some(g)=g {if g.shape!=input.shape {return Err(AutogradError::GradientShapeMismatch {expected:input.shape.clone(),got:g.shape.clone()}.into());}}}
            for (&id,g) in record.inputs.iter().zip(incoming) {if let Some(g)=g {
                if !state.tracked.contains(&id){continue;}
                if let Some(old)=state.gradients.get_mut(&id) {for (a,b) in old.data.iter_mut().zip(g.data) {*a+=b;}}
                else{state.gradients.insert(id,g);}
            }}
        }
        if !options.retain_graph {
            for record in &records {state.remove_graph(record.output)?;state.consumed.insert(record.output);}
            state.consumed.insert(root);state.collect()?;
        }
        let keep=state.leaves.union(&state.retained).copied().collect::<HashSet<_>>();
        state.gradients.retain(|id,_|keep.contains(id));Ok(())
    }
}
