use super::*;
use crate::{MlResult,MlError,Tensor,Variable};
use crate::trainer::TrainingRuntime;

pub struct Prebatched<B> { batches:Vec<B>, order:Vec<usize>, cursor:usize }
impl<B:Clone> BatchLoader for Prebatched<B> {
    type Batch=B;
    fn begin_epoch(&mut self,_epoch:usize,runtime:&TrainingRuntime)->MlResult<()> {
        self.order=(0..self.batches.len()).collect();runtime.shuffle(&mut self.order);self.cursor=0;Ok(())
    }
    fn next_batch(&mut self)->MlResult<Option<B>> {
        let Some(&index)=self.order.get(self.cursor) else{return Ok(None);};self.cursor+=1;Ok(Some(self.batches[index].clone()))
    }
    fn batch_count(&self)->Option<usize>{Some(self.batches.len())}
}
fn prebatched<B>(batches:Vec<B>)->Prebatched<B>{Prebatched {batches,order:Vec::new(),cursor:0}}
impl IntoBatchLoader for &SupervisedDataset<'_> {
    type Batch=SupervisedBatch;type Loader=Prebatched<Self::Batch>;
    fn into_batch_loader(self)->Self::Loader {prebatched(self.inputs.iter().zip(self.targets).map(|(x,t)|SupervisedBatch {inputs:(*x).clone(),targets:(*t).clone()}).collect())}
}
impl IntoBatchLoader for &UnsupervisedDataset<'_> {
    type Batch=UnsupervisedBatch;type Loader=Prebatched<Self::Batch>;
    fn into_batch_loader(self)->Self::Loader {prebatched(self.samples.iter().map(|x|UnsupervisedBatch {samples:(*x).clone()}).collect())}
}
impl IntoBatchLoader for &AutoregressiveDataset<'_> {
    type Batch=AutoregressiveBatch;type Loader=AutoregressivePrebatched;
    fn into_batch_loader(self)->Self::Loader {AutoregressivePrebatched {inner:prebatched(self.sequences.iter().map(|x|AutoregressiveBatch {sequences:(*x).clone()}).collect()),padding:self.pad_token_id}}
}
pub struct AutoregressivePrebatched {inner:Prebatched<AutoregressiveBatch>,padding:Option<usize>}
impl BatchLoader for AutoregressivePrebatched {
    type Batch=AutoregressiveBatch;
    fn begin_epoch(&mut self,epoch:usize,runtime:&TrainingRuntime)->MlResult<()> {
        if self.padding.is_some(){return Err(MlError::UnsupportedCapability {module:"autoregressive",capability:"padding",operation:"fit"});}
        self.inner.begin_epoch(epoch,runtime)
    }
    fn next_batch(&mut self)->MlResult<Option<Self::Batch>>{self.inner.next_batch()}
    fn batch_count(&self)->Option<usize>{self.inner.batch_count()}
}
impl IntoBatchLoader for &SemiSupervisedDataset<'_> {
    type Batch=SemiSupervisedBatch;type Loader=Prebatched<Self::Batch>;
    fn into_batch_loader(self)->Self::Loader {
        let count=self.labeled_inputs.len().max(self.unlabeled_inputs.len());
        prebatched((0..count).map(|i|SemiSupervisedBatch {
            labeled_inputs:self.labeled_inputs[i%self.labeled_inputs.len()].clone(),
            labeled_targets:self.labeled_targets[i%self.labeled_targets.len()].clone(),
            unlabeled_inputs:self.unlabeled_inputs[i%self.unlabeled_inputs.len()].clone(),
        }).collect())
    }
}
#[derive(Clone, Copy)]
pub struct SupervisedDataset<'a> { pub inputs: &'a [&'a Variable], pub targets: &'a [&'a Tensor] }
impl<'a> SupervisedDataset<'a> {
    pub fn new(_context: &crate::ExecutionContext, inputs: &'a [&'a Variable], targets: &'a [&'a Tensor]) -> MlResult<Self> {
        if inputs.is_empty() { return Err(MlError::StringError("supervised dataset must not be empty".into())); }
        if inputs.len() != targets.len() { return Err(MlError::StringError("input/target length mismatch".into())); }
        Ok(Self { inputs, targets })
    }
}

#[derive(Clone, Copy)]
pub struct UnsupervisedDataset<'a> { pub samples: &'a [&'a Variable] }
impl<'a> UnsupervisedDataset<'a> {
    pub fn new(_context: &crate::ExecutionContext, samples: &'a [&'a Variable]) -> MlResult<Self> {
        if samples.is_empty() { return Err(MlError::StringError("unsupervised dataset must not be empty".into())); }
        Ok(Self { samples })
    }
}

#[derive(Clone, Copy)]
pub struct SemiSupervisedDataset<'a> {
    pub labeled_inputs: &'a [&'a Variable], pub labeled_targets: &'a [&'a Tensor],
    pub unlabeled_inputs: &'a [&'a Variable],
}
impl<'a> SemiSupervisedDataset<'a> {
    pub fn new(_context: &crate::ExecutionContext, labeled_inputs: &'a [&'a Variable], labeled_targets: &'a [&'a Tensor],
               unlabeled_inputs: &'a [&'a Variable]) -> MlResult<Self> {
        if labeled_inputs.is_empty() || unlabeled_inputs.is_empty() {
            return Err(MlError::StringError("semi-supervised datasets must not be empty".into()));
        }
        if labeled_inputs.len() != labeled_targets.len() {
            return Err(MlError::StringError("labeled input/target length mismatch".into()));
        }
        Ok(Self { labeled_inputs, labeled_targets, unlabeled_inputs })
    }
}

#[derive(Clone, Copy)]
pub struct AutoregressiveDataset<'a> { pub sequences: &'a [&'a Variable], pub pad_token_id: Option<usize> }
impl<'a> AutoregressiveDataset<'a> {
    pub fn new(_context: &crate::ExecutionContext, sequences: &'a [&'a Variable]) -> MlResult<Self> {
        if sequences.is_empty() { return Err(MlError::StringError("autoregressive dataset must not be empty".into())); }
        Ok(Self { sequences, pad_token_id: None })
    }
    pub fn with_pad_token_id(mut self, id: usize) -> Self { self.pad_token_id = Some(id); self }
}

