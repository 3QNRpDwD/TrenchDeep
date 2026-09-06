use super::*;
use crate::optimizer::{Optimizer,clip_context_grad_norm};
use std::{collections::HashSet,time::{Instant,Duration},path::Path};

pub(super) struct StepData {
    pub loss:Variable,pub prediction:Option<Variable>,pub target:Option<Tensor>,
    pub weight:usize,pub tokens:Option<usize>,pub lambda:Option<f32>,
}
pub(super) struct StepOutcome {pub loss:f32,pub weight:usize,pub metrics:MetricValues}
pub(super) trait BatchInputs {fn tensors(&self)->Vec<&Tensor>;}
impl BatchInputs for SupervisedBatch {fn tensors(&self)->Vec<&Tensor>{vec![self.inputs.tensor(),&self.targets]}}
impl BatchInputs for UnsupervisedBatch {fn tensors(&self)->Vec<&Tensor>{vec![self.samples.tensor()]}}
impl BatchInputs for AutoregressiveBatch {fn tensors(&self)->Vec<&Tensor>{vec![self.sequences.tensor()]}}
impl BatchInputs for SemiSupervisedBatch {fn tensors(&self)->Vec<&Tensor>{vec![self.labeled_inputs.tensor(),&self.labeled_targets,self.unlabeled_inputs.tensor()]}}

pub(super) fn validate_parameters<M:TrainableModel+?Sized>(ctx:&ExecutionContext,model:&M,optimizer:&dyn Optimizer)->MlResult<()> {
    if model.context_id()!=ctx.id() || optimizer.context_id()!=ctx.id(){return Err(ContextError::Mismatch.into());}
    let expected=model.parameters();let actual=optimizer.registered_parameters();
    for p in expected.iter().chain(&actual){ctx.validate(p.tensor())?;}
    let expected=expected.iter().map(|p|p.id()).collect::<HashSet<_>>();
    let actual=actual.iter().map(|p|p.id()).collect::<HashSet<_>>();
    if expected!=actual {return Err(crate::OptimError::ParameterSetMismatch {missing:expected.difference(&actual).copied().collect(),extra:actual.difference(&expected).copied().collect()}.into());}
    Ok(())
}
pub(super) fn sample_count(input:&Tensor)->MlResult<usize>{let shape=input.shape()?;Ok(if shape.len()>1 {shape[0]}else{1})}

pub(super) struct TrainingService {pub context:ExecutionContext,pub core:TrainerCore,pub max_grad_norm:Option<f32>}
impl TrainingService {
    pub fn new(context:&ExecutionContext,trainer:Trainer)->Self {Self {context:context.clone(),core:trainer.core,max_grad_norm:None}}
    /// The caller owns the scope so loading, rollout and forward failures share cleanup.
    pub fn finish_step<M:TrainableModel>(&self,model:&M,optimizer:&mut dyn Optimizer,data:StepData,batch:&BatchStartContext,forward:Duration)->MlResult<StepOutcome> {
        validate_parameters(&self.context,model,optimizer)?;
        self.context.validate(data.loss.tensor())?;
        if let Some(p)=&data.prediction {self.context.validate(p.tensor())?;}
        if let Some(t)=&data.target {self.context.validate(t)?;}
        let loss=data.loss.tensor().item()?;
        if !loss.is_finite(){return Err(crate::TensorError::InvalidOperation {op:"training",reason:"non-finite loss".into()}.into());}
        if data.weight==0 {return Err(DataError::NoBatches.into());}
        let backward_start=Instant::now();data.loss.backward()?;let backward=backward_start.elapsed();
        let parameters=model.parameters();let mut metrics=MetricValues::new();
        let mut squared=0.0;let mut weights=0.0;
        let check=self.core.config.nan_check_interval!=usize::MAX && batch.batch%self.core.config.nan_check_interval==0;
        for p in &parameters {if let Some(g)=p.grad()? {
            if check && g.data().iter().any(|x|!x.is_finite()){return Err(crate::TensorError::InvalidOperation {op:"training",reason:"non-finite gradient".into()}.into());}
            squared+=g.data().iter().map(|x|x*x).sum::<f32>();
        } if self.core.config.metrics.update_ratio {weights+=p.tensor().with_view(|v|v.data().iter().map(|x|x*x).sum::<f32>())?;}}
        if self.core.config.metrics.grad_norm {metrics.insert("grad_norm".into(),squared.sqrt());}
        if self.core.config.metrics.update_ratio {metrics.insert("update_ratio".into(),if weights>1e-12 {optimizer.lr()*squared.sqrt()/weights.sqrt()}else{0.0});}
        if self.core.config.metrics.fw_bw_timing {metrics.insert("forward_secs".into(),forward.as_secs_f32());metrics.insert("backward_secs".into(),backward.as_secs_f32());}
        if let Some(max)=self.max_grad_norm {clip_context_grad_norm(&self.context,&parameters,max)?;}
        if self.core.hook_count()!=0 {
            let pred=data.prediction.as_ref().map(|p|p.tensor().snapshot()).transpose()?;
            let target=data.target.as_ref().map(Tensor::snapshot).transpose()?;
            let observation=BatchContext {batch_idx:batch.batch-1,pred:pred.as_ref(),target:target.as_ref(),loss,n_tokens:data.tokens,lambda:data.lambda,lr:optimizer.lr()};
            for hook in self.core.hooks.borrow_mut().iter_mut(){hook.update(&observation)?;}
        }
        optimizer.step()?;optimizer.zero_grad()?;
        Ok(StepOutcome {loss,weight:data.weight,metrics})
    }
    pub fn fit<M,I,F>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule,paradigm:&'static str,mut forward:F,save:Option<fn(&M,&Path)->MlResult<()>>)->MlResult<TrainResult>
    where M:TrainableModel,I:IntoBatchLoader,I::Batch:BatchInputs,F:FnMut(&mut M,I::Batch,usize)->MlResult<StepData> {
        validate_parameters(&self.context,model,optimizer)?;
        if self.core.config.checkpoint_dir.is_some() && save.is_none(){return Err(MlError::UnsupportedCapability {module:"trainer",capability:"checkpointing requires fit_checkpointed",operation:"fit"});}
        // Reject pre-existing graphs before calling a loader or a model.
        self.context.begin_training_scope()?.finish(Ok(()))?;
        let started=Instant::now();let mut loader=input.into_batch_loader();
        self.core.notify_train_start(&TrainStartContext {paradigm,total_units:schedule.epochs});
        let progress=progress::EpochProgress::new(schedule.epochs,self.core.config.show_progress);
        let result:MlResult<TrainResult>=(|| {
            let mut previous=f32::INFINITY;let mut final_loss=f32::INFINITY;let mut final_metrics=MetricValues::new();let mut completed=0;let mut reason=StopReason::Completed;
            for epoch in 0..schedule.epochs {
                self.core.begin_epoch(epoch);loader.begin_epoch(epoch,&self.core.runtime)?;
                for hook in self.core.hooks.borrow_mut().iter_mut(){hook.reset()?;}
                let epoch_context=EpochContext {paradigm,epoch:epoch+1,total_epochs:schedule.epochs,total_batches:loader.batch_count()};
                self.core.notify_epoch_start(&epoch_context);let epoch_start=Instant::now();
                let batch_progress=progress.start_batch_bar(epoch,schedule.epochs,loader.batch_count());
                let mut sum=0.0;let mut weight=0usize;let mut batches=0;let mut metrics=MetricValues::new();
                loop {
                    let scope=self.context.begin_training_scope()?;
                    let batch_context=BatchStartContext {paradigm,epoch:epoch+1,batch:batches+1,total_epochs:schedule.epochs,total_batches:loader.batch_count(),episode:None};
                    let batch=(|| {
                        let Some(batch)=loader.next_batch()? else{return Ok(None);};
                        for tensor in batch.tensors(){self.context.validate(tensor)?;}
                        let start=Instant::now();let data=forward(model,batch,epoch)?;
                        self.finish_step(model,optimizer,data,&batch_context,start.elapsed()).map(Some)
                    })();
                    let Some(outcome)=scope.finish(batch)? else{break;};
                    self.core.notify_batch_end(&BatchEndContext {batch:batch_context,loss:outcome.loss});
                    sum+=outcome.loss*outcome.weight as f32;weight+=outcome.weight;batches+=1;
                    for (key,value) in outcome.metrics {*metrics.entry(key).or_insert(0.0)+=value;}
                    batch_progress.inc();
                    if self.core.config.batch_log_interval!=usize::MAX && batches%self.core.config.batch_log_interval==0 {batch_progress.set_msg(&format!("loss: {:.6}",outcome.loss));}
                    if checkpoint::interrupted(){reason=StopReason::Interrupted;break;}
                }
                batch_progress.finish();
                if weight==0{return Err(DataError::NoBatches.into());}
                final_loss=sum/weight as f32;
                for value in metrics.values_mut(){*value/=batches as f32;}
                metrics.insert("avg_loss".into(),final_loss);metrics.insert("epoch_duration_secs".into(),epoch_start.elapsed().as_secs_f32());
                for hook in self.core.hooks.borrow().iter(){metrics.insert(hook.name().into(),hook.compute());}
                final_metrics=metrics;
                if reason!=StopReason::Interrupted {completed=epoch+1;}
                self.core.notify_epoch_end(&epoch_context);progress.inc();
                if self.core.config.epoch_log_interval!=usize::MAX && (epoch+1)%self.core.config.epoch_log_interval==0 {tracing::info!(paradigm,epoch=epoch+1,loss=final_loss,"training epoch");}
                if reason==StopReason::Interrupted {break;}
                if schedule.convergence.should_stop(previous,final_loss){reason=StopReason::Converged;break;}previous=final_loss;
            }
            let mut result=TrainResult::epochs(reason,completed,final_loss,started.elapsed()).with_metrics(final_metrics);
            if reason==StopReason::Interrupted {if let (Some(directory),Some(save))=(&self.core.config.checkpoint_dir,save) {
                result.checkpoint=Some(checkpoint::save_model(directory,paradigm,completed,schedule,final_loss,optimizer.lr(),self.core.config.seed,|path|save(model,path))?);
            }}
            self.core.notify_train_end(&TrainEndContext {paradigm,units_completed:completed,interrupted:reason==StopReason::Interrupted});Ok(result)
        })();
        match &result {Ok(_)=>progress.finish_completed(),Err(e)=>{progress.abandon("training failed");self.core.notify_train_error(&e.to_string());}}
        result
    }
}
