use super::*;
use super::service::{TrainingService,StepData,sample_count};
use crate::optimizer::Optimizer;
pub struct SupervisedTrainer { service:TrainingService }
impl SupervisedTrainer {
    pub fn new(context:&ExecutionContext)->Self {Self::silent(context)}
    pub fn from_trainer(context:&ExecutionContext,trainer:Trainer)->Self { Self {service:TrainingService::new(context,trainer)} }
    pub fn silent(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::silent())}
    pub fn minimal(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::minimal())}
    pub fn default(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::default())}
    pub fn verbose(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::verbose())}
    pub fn with_seed(mut self,seed:u64)->Self {self.service.core.config.seed=seed;self.service.core.runtime.reseed(seed);self}
    pub fn with_hook(self,hook:Box<dyn MetricHook>)->Self {self.service.core.add_hook(hook);self}
    pub fn with_observer(self,observer:Box<dyn TrainingObserver>)->Self {self.service.core.add_observer(observer);self}
    pub fn check_finite_gradients(mut self,enabled:bool)->Self {self.service.core.config.nan_check_interval=if enabled {1}else{usize::MAX};self}
    pub fn with_max_grad_norm(mut self,max:f32)->MlResult<Self> {if !max.is_finite() || max<=0.0 {return Err(MlError::StringError("max_grad_norm must be finite and positive".into()));}self.service.max_grad_norm=Some(max);Ok(self)}
    pub fn fit<M:SupervisedModel,I:IntoBatchLoader<Batch=SupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        self.service.fit(model,optimizer,input,schedule,"supervised",|model,batch,epoch|{let weight=sample_count(batch.inputs.tensor())?;let (prediction,loss)=model.forward_loss(&batch.inputs,&batch.targets)?;
Ok(StepData {loss,prediction:Some(prediction),target:Some(batch.targets),weight,tokens:None,lambda:None})},None)
    }
    pub fn fit_loader<M:SupervisedModel,I:IntoBatchLoader<Batch=SupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {self.fit(model,optimizer,input,schedule)}
    pub fn fit_checkpointed<M:SupervisedModel+CheckpointableModel,I:IntoBatchLoader<Batch=SupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        checkpoint::install_interrupt_handler()?;
        self.service.fit(model,optimizer,input,schedule,"supervised",|model,batch,epoch|{let weight=sample_count(batch.inputs.tensor())?;let (prediction,loss)=model.forward_loss(&batch.inputs,&batch.targets)?;
Ok(StepData {loss,prediction:Some(prediction),target:Some(batch.targets),weight,tokens:None,lambda:None})},Some(|model,path|model.save_checkpoint(path)))
    }
    pub fn resume<M:SupervisedModel,I:IntoBatchLoader<Batch=SupervisedBatch>>(&self,_model:&mut M,_optimizer:&mut dyn Optimizer,_input:I,_path:&str,_schedule:EpochSchedule)->MlResult<TrainResult> {Err(MlError::UnsupportedCapability {module:"trainer",capability:"complete resume (P2)",operation:"resume"})}
}
pub struct UnsupervisedTrainer { service:TrainingService }
impl UnsupervisedTrainer {
    pub fn new(context:&ExecutionContext)->Self {Self::silent(context)}
    pub fn from_trainer(context:&ExecutionContext,trainer:Trainer)->Self { Self {service:TrainingService::new(context,trainer)} }
    pub fn silent(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::silent())}
    pub fn minimal(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::minimal())}
    pub fn default(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::default())}
    pub fn verbose(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::verbose())}
    pub fn with_seed(mut self,seed:u64)->Self {self.service.core.config.seed=seed;self.service.core.runtime.reseed(seed);self}
    pub fn with_hook(self,hook:Box<dyn MetricHook>)->Self {self.service.core.add_hook(hook);self}
    pub fn with_observer(self,observer:Box<dyn TrainingObserver>)->Self {self.service.core.add_observer(observer);self}
    pub fn check_finite_gradients(mut self,enabled:bool)->Self {self.service.core.config.nan_check_interval=if enabled {1}else{usize::MAX};self}
    pub fn with_max_grad_norm(mut self,max:f32)->MlResult<Self> {if !max.is_finite() || max<=0.0 {return Err(MlError::StringError("max_grad_norm must be finite and positive".into()));}self.service.max_grad_norm=Some(max);Ok(self)}
    pub fn fit<M:UnsupervisedModel,I:IntoBatchLoader<Batch=UnsupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        self.service.fit(model,optimizer,input,schedule,"unsupervised",|model,batch,epoch|{let weight=sample_count(batch.samples.tensor())?;let (prediction,loss)=model.forward_loss(&batch.samples)?;
Ok(StepData {loss,prediction:Some(prediction),target:None,weight,tokens:None,lambda:None})},None)
    }
    pub fn fit_loader<M:UnsupervisedModel,I:IntoBatchLoader<Batch=UnsupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {self.fit(model,optimizer,input,schedule)}
    pub fn fit_checkpointed<M:UnsupervisedModel+CheckpointableModel,I:IntoBatchLoader<Batch=UnsupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        checkpoint::install_interrupt_handler()?;
        self.service.fit(model,optimizer,input,schedule,"unsupervised",|model,batch,epoch|{let weight=sample_count(batch.samples.tensor())?;let (prediction,loss)=model.forward_loss(&batch.samples)?;
Ok(StepData {loss,prediction:Some(prediction),target:None,weight,tokens:None,lambda:None})},Some(|model,path|model.save_checkpoint(path)))
    }
    pub fn resume<M:UnsupervisedModel,I:IntoBatchLoader<Batch=UnsupervisedBatch>>(&self,_model:&mut M,_optimizer:&mut dyn Optimizer,_input:I,_path:&str,_schedule:EpochSchedule)->MlResult<TrainResult> {Err(MlError::UnsupportedCapability {module:"trainer",capability:"complete resume (P2)",operation:"resume"})}
}
pub struct AutoregressiveTrainer { service:TrainingService }
impl AutoregressiveTrainer {
    pub fn new(context:&ExecutionContext)->Self {Self::silent(context)}
    pub fn from_trainer(context:&ExecutionContext,trainer:Trainer)->Self { Self {service:TrainingService::new(context,trainer)} }
    pub fn silent(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::silent())}
    pub fn minimal(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::minimal())}
    pub fn default(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::default())}
    pub fn verbose(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::verbose())}
    pub fn with_seed(mut self,seed:u64)->Self {self.service.core.config.seed=seed;self.service.core.runtime.reseed(seed);self}
    pub fn with_hook(self,hook:Box<dyn MetricHook>)->Self {self.service.core.add_hook(hook);self}
    pub fn with_observer(self,observer:Box<dyn TrainingObserver>)->Self {self.service.core.add_observer(observer);self}
    pub fn check_finite_gradients(mut self,enabled:bool)->Self {self.service.core.config.nan_check_interval=if enabled {1}else{usize::MAX};self}
    pub fn with_max_grad_norm(mut self,max:f32)->MlResult<Self> {if !max.is_finite() || max<=0.0 {return Err(MlError::StringError("max_grad_norm must be finite and positive".into()));}self.service.max_grad_norm=Some(max);Ok(self)}
    pub fn fit<M:AutoregressiveModel,I:IntoBatchLoader<Batch=AutoregressiveBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        self.service.fit(model,optimizer,input,schedule,"autoregressive",|model,batch,epoch|{let (prediction,loss,tokens)=model.forward_loss(&batch.sequences)?;
Ok(StepData {loss,prediction:Some(prediction),target:None,weight:tokens,tokens:Some(tokens),lambda:None})},None)
    }
    pub fn fit_loader<M:AutoregressiveModel,I:IntoBatchLoader<Batch=AutoregressiveBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {self.fit(model,optimizer,input,schedule)}
    pub fn fit_checkpointed<M:AutoregressiveModel+CheckpointableModel,I:IntoBatchLoader<Batch=AutoregressiveBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        checkpoint::install_interrupt_handler()?;
        self.service.fit(model,optimizer,input,schedule,"autoregressive",|model,batch,epoch|{let (prediction,loss,tokens)=model.forward_loss(&batch.sequences)?;
Ok(StepData {loss,prediction:Some(prediction),target:None,weight:tokens,tokens:Some(tokens),lambda:None})},Some(|model,path|model.save_checkpoint(path)))
    }
    pub fn resume<M:AutoregressiveModel,I:IntoBatchLoader<Batch=AutoregressiveBatch>>(&self,_model:&mut M,_optimizer:&mut dyn Optimizer,_input:I,_path:&str,_schedule:EpochSchedule)->MlResult<TrainResult> {Err(MlError::UnsupportedCapability {module:"trainer",capability:"complete resume (P2)",operation:"resume"})}
}
pub struct SemiSupervisedTrainer { service:TrainingService,ramp:ConsistencyRamp }
impl SemiSupervisedTrainer {
    pub fn new(context:&ExecutionContext)->Self {Self::silent(context)}
    pub fn from_trainer(context:&ExecutionContext,trainer:Trainer)->Self { Self {service:TrainingService::new(context,trainer),ramp:ConsistencyRamp::default()} }
    pub fn silent(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::silent())}
    pub fn minimal(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::minimal())}
    pub fn default(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::default())}
    pub fn verbose(context:&ExecutionContext)->Self {Self::from_trainer(context,Trainer::verbose())}
    pub fn with_seed(mut self,seed:u64)->Self {self.service.core.config.seed=seed;self.service.core.runtime.reseed(seed);self}
    pub fn with_hook(self,hook:Box<dyn MetricHook>)->Self {self.service.core.add_hook(hook);self}
    pub fn with_observer(self,observer:Box<dyn TrainingObserver>)->Self {self.service.core.add_observer(observer);self}
    pub fn check_finite_gradients(mut self,enabled:bool)->Self {self.service.core.config.nan_check_interval=if enabled {1}else{usize::MAX};self}
    pub fn with_max_grad_norm(mut self,max:f32)->MlResult<Self> {if !max.is_finite() || max<=0.0 {return Err(MlError::StringError("max_grad_norm must be finite and positive".into()));}self.service.max_grad_norm=Some(max);Ok(self)}
    pub fn fit<M:SemiSupervisedModel,I:IntoBatchLoader<Batch=SemiSupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        self.service.fit(model,optimizer,input,schedule,"semi_supervised",|model,batch,epoch|{let weight=sample_count(batch.labeled_inputs.tensor())?;let lambda=self.ramp.value(epoch);
let (prediction,loss)=model.forward_loss(&batch.labeled_inputs,&batch.labeled_targets,&batch.unlabeled_inputs,lambda)?;
Ok(StepData {loss,prediction:Some(prediction),target:Some(batch.labeled_targets),weight,tokens:None,lambda:Some(lambda)})},None)
    }
    pub fn fit_loader<M:SemiSupervisedModel,I:IntoBatchLoader<Batch=SemiSupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {self.fit(model,optimizer,input,schedule)}
    pub fn fit_checkpointed<M:SemiSupervisedModel+CheckpointableModel,I:IntoBatchLoader<Batch=SemiSupervisedBatch>>(&self,model:&mut M,optimizer:&mut dyn Optimizer,input:I,schedule:EpochSchedule)->MlResult<TrainResult> {
        checkpoint::install_interrupt_handler()?;
        self.service.fit(model,optimizer,input,schedule,"semi_supervised",|model,batch,epoch|{let weight=sample_count(batch.labeled_inputs.tensor())?;let lambda=self.ramp.value(epoch);
let (prediction,loss)=model.forward_loss(&batch.labeled_inputs,&batch.labeled_targets,&batch.unlabeled_inputs,lambda)?;
Ok(StepData {loss,prediction:Some(prediction),target:Some(batch.labeled_targets),weight,tokens:None,lambda:Some(lambda)})},Some(|model,path|model.save_checkpoint(path)))
    }
    pub fn resume<M:SemiSupervisedModel,I:IntoBatchLoader<Batch=SemiSupervisedBatch>>(&self,_model:&mut M,_optimizer:&mut dyn Optimizer,_input:I,_path:&str,_schedule:EpochSchedule)->MlResult<TrainResult> {Err(MlError::UnsupportedCapability {module:"trainer",capability:"complete resume (P2)",operation:"resume"})}
    pub fn with_ramp(mut self,ramp:ConsistencyRamp)->Self {self.ramp=ramp;self}
}
