use crate::{
    MlResult,
    nn::Variable,
    trainer::{
        AutoregressiveDataset, SemiSupervisedDataset, SupervisedDataset, TrainingRuntime,
        UnsupervisedDataset,
    },
};

use super::{
    AutoregressiveBatch, BatchLoader, IntoBatchLoader, SemiSupervisedBatch, SupervisedBatch,
    UnsupervisedBatch,
};

pub struct PreBatchedSupervised<'a> {
    dataset: SupervisedDataset<'a>,
    order: Vec<usize>,
    cursor: usize,
}
pub struct PreBatchedUnsupervised<'a> {
    dataset: UnsupervisedDataset<'a>,
    order: Vec<usize>,
    cursor: usize,
}
pub struct PreBatchedAutoregressive<'a> {
    dataset: AutoregressiveDataset<'a>,
    order: Vec<usize>,
    cursor: usize,
}
pub struct PreBatchedSemiSupervised<'a> {
    dataset: SemiSupervisedDataset<'a>,
    labeled: Vec<usize>,
    unlabeled: Vec<usize>,
    cursor: usize,
}

impl<'a> IntoBatchLoader for SupervisedDataset<'a> {
    type Batch = SupervisedBatch;
    type Loader = PreBatchedSupervised<'a>;
    fn into_batch_loader(self) -> Self::Loader {
        PreBatchedSupervised {
            order: (0..self.inputs.len()).collect(),
            dataset: self,
            cursor: 0,
        }
    }
}
impl<'a> IntoBatchLoader for UnsupervisedDataset<'a> {
    type Batch = UnsupervisedBatch;
    type Loader = PreBatchedUnsupervised<'a>;
    fn into_batch_loader(self) -> Self::Loader {
        PreBatchedUnsupervised {
            order: (0..self.samples.len()).collect(),
            dataset: self,
            cursor: 0,
        }
    }
}
impl<'a> IntoBatchLoader for AutoregressiveDataset<'a> {
    type Batch = AutoregressiveBatch;
    type Loader = PreBatchedAutoregressive<'a>;
    fn into_batch_loader(self) -> Self::Loader {
        PreBatchedAutoregressive {
            order: (0..self.sequences.len()).collect(),
            dataset: self,
            cursor: 0,
        }
    }
}
impl<'a> IntoBatchLoader for SemiSupervisedDataset<'a> {
    type Batch = SemiSupervisedBatch;
    type Loader = PreBatchedSemiSupervised<'a>;
    fn into_batch_loader(self) -> Self::Loader {
        PreBatchedSemiSupervised {
            labeled: (0..self.labeled_inputs.len()).collect(),
            unlabeled: (0..self.unlabeled_inputs.len()).collect(),
            dataset: self,
            cursor: 0,
        }
    }
}

impl BatchLoader for PreBatchedSupervised<'_> {
    type Batch = SupervisedBatch;
    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        self.order.clear();
        self.order.extend(0..self.dataset.inputs.len());
        runtime.shuffle(&mut self.order);
        self.cursor = 0;
        Ok(())
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        let Some(&i) = self.order.get(self.cursor) else {
            return Ok(None);
        };
        self.cursor += 1;
        Ok(Some(SupervisedBatch {
            inputs: self.dataset.inputs[i].clone(),
            targets: self.dataset.targets[i].clone(),
        }))
    }
    fn batch_count(&self) -> Option<usize> {
        Some(self.dataset.inputs.len())
    }
}
impl BatchLoader for PreBatchedUnsupervised<'_> {
    type Batch = UnsupervisedBatch;
    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        self.order.clear();
        self.order.extend(0..self.dataset.samples.len());
        runtime.shuffle(&mut self.order);
        self.cursor = 0;
        Ok(())
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        let Some(&i) = self.order.get(self.cursor) else {
            return Ok(None);
        };
        self.cursor += 1;
        Ok(Some(UnsupervisedBatch {
            samples: self.dataset.samples[i].clone(),
        }))
    }
    fn batch_count(&self) -> Option<usize> {
        Some(self.dataset.samples.len())
    }
}
impl BatchLoader for PreBatchedAutoregressive<'_> {
    type Batch = AutoregressiveBatch;
    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        self.order.clear();
        self.order.extend(0..self.dataset.sequences.len());
        runtime.shuffle(&mut self.order);
        self.cursor = 0;
        Ok(())
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        let Some(&i) = self.order.get(self.cursor) else {
            return Ok(None);
        };
        self.cursor += 1;
        Ok(Some(AutoregressiveBatch {
            sequences: self.dataset.sequences[i].clone(),
        }))
    }
    fn batch_count(&self) -> Option<usize> {
        Some(self.dataset.sequences.len())
    }
}
impl BatchLoader for PreBatchedSemiSupervised<'_> {
    type Batch = SemiSupervisedBatch;
    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        self.labeled.clear();
        self.labeled.extend(0..self.dataset.labeled_inputs.len());
        self.unlabeled.clear();
        self.unlabeled
            .extend(0..self.dataset.unlabeled_inputs.len());
        runtime.shuffle(&mut self.labeled);
        runtime.shuffle(&mut self.unlabeled);
        self.cursor = 0;
        Ok(())
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        let total = self.labeled.len().max(self.unlabeled.len());
        if self.cursor >= total {
            return Ok(None);
        }
        let li = self.labeled[self.cursor % self.labeled.len()];
        let ui = self.unlabeled[self.cursor % self.unlabeled.len()];
        self.cursor += 1;
        Ok(Some(SemiSupervisedBatch {
            labeled_inputs: self.dataset.labeled_inputs[li].clone(),
            labeled_targets: self.dataset.labeled_targets[li].clone(),
            unlabeled_inputs: self.dataset.unlabeled_inputs[ui].clone(),
        }))
    }
    fn batch_count(&self) -> Option<usize> {
        Some(self.labeled.len().max(self.unlabeled.len()))
    }
}

#[allow(dead_code)]
fn _variable_is_clone(_: Variable) {}
