use crate::trainer::TrainingRuntime;
use crate::{
    MlError, MlResult,
    nn::{Parameter, Variable},
    tensor::{Tensor, TensorBase},
};

use super::DataError;

fn classify_collate_error(error: MlError) -> MlError {
    match error {
        shape @ MlError::DataError(DataError::ShapeMismatch { .. }) => shape,
        empty @ MlError::DataError(DataError::EmptyBatch) => empty,
        other => DataError::Collate {
            message: other.to_string(),
        }
        .into(),
    }
}

pub trait Dataset {
    type Sample;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&Self::Sample>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug)]
pub struct InMemoryDataset<S> {
    samples: Vec<S>,
}

impl<S> InMemoryDataset<S> {
    pub fn new(samples: Vec<S>) -> Result<Self, DataError> {
        if samples.is_empty() {
            return Err(DataError::EmptyDataset);
        }
        Ok(Self { samples })
    }
    pub fn samples(&self) -> &[S] {
        &self.samples
    }
    pub fn into_samples(self) -> Vec<S> {
        self.samples
    }
}

impl<S> Dataset for InMemoryDataset<S> {
    type Sample = S;
    fn len(&self) -> usize {
        self.samples.len()
    }
    fn get(&self, index: usize) -> Option<&S> {
        self.samples.get(index)
    }
}

pub trait Collator<S> {
    type Batch;
    fn collate(&mut self, samples: &[&S]) -> MlResult<Self::Batch>;
}

impl<S, B, F> Collator<S> for F
where
    F: FnMut(&[&S]) -> MlResult<B>,
{
    type Batch = B;
    fn collate(&mut self, samples: &[&S]) -> MlResult<B> {
        self(samples)
    }
}

pub trait BatchLoader {
    type Batch;
    fn begin_epoch(&mut self, epoch: usize, runtime: &TrainingRuntime) -> MlResult<()>;
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>>;
    fn batch_count(&self) -> Option<usize>;
}

impl<L: BatchLoader + ?Sized> BatchLoader for &mut L {
    type Batch = L::Batch;
    fn begin_epoch(&mut self, epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        (**self).begin_epoch(epoch, runtime)
    }
    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        (**self).next_batch()
    }
    fn batch_count(&self) -> Option<usize> {
        (**self).batch_count()
    }
}

pub trait IntoBatchLoader {
    type Batch;
    type Loader: BatchLoader<Batch = Self::Batch>;
    fn into_batch_loader(self) -> Self::Loader;
}

impl<L: BatchLoader> IntoBatchLoader for L {
    type Batch = L::Batch;
    type Loader = L;
    fn into_batch_loader(self) -> L {
        self
    }
}

pub struct DataLoader<D, C> {
    dataset: D,
    collator: C,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    indices: Vec<usize>,
    cursor: usize,
}

#[doc(hidden)]
pub struct MissingCollator;

pub struct DataLoaderBuilder<D, C = MissingCollator> {
    dataset: D,
    collator: C,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<D> DataLoader<D, MissingCollator> {
    pub fn builder(dataset: D) -> DataLoaderBuilder<D> {
        DataLoaderBuilder {
            dataset,
            collator: MissingCollator,
            batch_size: 1,
            shuffle: true,
            drop_last: false,
        }
    }
}

impl<D, C> DataLoaderBuilder<D, C> {
    pub fn collator<C2>(self, collator: C2) -> DataLoaderBuilder<D, C2> {
        DataLoaderBuilder {
            dataset: self.dataset,
            collator,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
        }
    }
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
}

impl<D, C> DataLoaderBuilder<D, C>
where
    D: Dataset,
    C: Collator<D::Sample>,
{
    pub fn build(self) -> MlResult<DataLoader<D, C>> {
        if self.dataset.is_empty() {
            return Err(DataError::EmptyDataset.into());
        }
        if self.batch_size == 0 {
            return Err(DataError::InvalidBatchSize.into());
        }
        if self.drop_last && self.dataset.len() < self.batch_size {
            return Err(DataError::NoBatches.into());
        }
        let indices = (0..self.dataset.len()).collect();
        Ok(DataLoader {
            dataset: self.dataset,
            collator: self.collator,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            indices,
            cursor: 0,
        })
    }
}

impl<D, C> DataLoader<D, C>
where
    D: Dataset,
    C: Collator<D::Sample>,
{
    fn count(&self) -> usize {
        if self.drop_last {
            self.dataset.len() / self.batch_size
        } else {
            self.dataset.len().div_ceil(self.batch_size)
        }
    }
}

impl<D, C> BatchLoader for DataLoader<D, C>
where
    D: Dataset,
    C: Collator<D::Sample>,
{
    type Batch = C::Batch;

    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        self.indices.clear();
        self.indices.extend(0..self.dataset.len());
        if self.shuffle {
            runtime.shuffle(&mut self.indices);
        }
        self.cursor = 0;
        Ok(())
    }

    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        if self.cursor >= self.indices.len() {
            return Ok(None);
        }
        let end = (self.cursor + self.batch_size).min(self.indices.len());
        if self.drop_last && end - self.cursor < self.batch_size {
            self.cursor = self.indices.len();
            return Ok(None);
        }
        let samples = self.indices[self.cursor..end]
            .iter()
            .map(|&index| self.dataset.get(index).expect("loader index must be valid"))
            .collect::<Vec<_>>();
        self.cursor = end;
        self.collator
            .collate(&samples)
            .map(Some)
            .map_err(classify_collate_error)
    }

    fn batch_count(&self) -> Option<usize> {
        Some(self.count())
    }
}

#[derive(Debug, Clone)]
pub struct SupervisedSample {
    pub input: Tensor,
    pub target: Tensor,
}
impl SupervisedSample {
    pub fn new(input: Tensor, target: Tensor) -> Self {
        Self { input, target }
    }
}

#[derive(Debug, Clone)]
pub struct UnsupervisedSample {
    pub input: Tensor,
}
impl UnsupervisedSample {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
}

#[derive(Debug, Clone)]
pub struct AutoregressiveSample {
    pub sequence: Tensor,
}
impl AutoregressiveSample {
    pub fn new(sequence: Tensor) -> Self {
        Self { sequence }
    }
}

#[derive(Debug)]
pub struct SupervisedBatch {
    pub inputs: Variable,
    pub targets: Variable,
}
#[derive(Debug)]
pub struct UnsupervisedBatch {
    pub samples: Variable,
}
#[derive(Debug)]
pub struct AutoregressiveBatch {
    pub sequences: Variable,
}
#[derive(Debug)]
pub struct SemiSupervisedBatch {
    pub labeled_inputs: Variable,
    pub labeled_targets: Variable,
    pub unlabeled_inputs: Variable,
}

fn stack_tensors(tensors: &[&Tensor]) -> MlResult<Tensor> {
    let Some(first) = tensors.first() else {
        return Err(DataError::EmptyBatch.into());
    };
    let expected = first.shape().to_vec();
    let mut data = Vec::with_capacity(tensors.len() * first.data().len());
    for (sample_index, tensor) in tensors.iter().enumerate() {
        if tensor.shape() != expected {
            return Err(DataError::ShapeMismatch {
                sample_index,
                expected: expected.clone(),
                got: tensor.shape().to_vec(),
            }
            .into());
        }
        data.extend_from_slice(tensor.data());
    }
    let mut shape = Vec::with_capacity(expected.len() + 1);
    shape.push(tensors.len());
    shape.extend(expected);
    Tensor::from_vec(data, &shape)
}

#[derive(Default)]
pub struct SupervisedStackCollator;
impl SupervisedStackCollator {
    pub fn new() -> Self {
        Self
    }
}
impl Collator<SupervisedSample> for SupervisedStackCollator {
    type Batch = SupervisedBatch;
    fn collate(&mut self, samples: &[&SupervisedSample]) -> MlResult<Self::Batch> {
        let inputs = samples
            .iter()
            .map(|sample| &sample.input)
            .collect::<Vec<_>>();
        let targets = samples
            .iter()
            .map(|sample| &sample.target)
            .collect::<Vec<_>>();
        Ok(SupervisedBatch {
            inputs: Variable::new(stack_tensors(&inputs)?),
            targets: Variable::new(stack_tensors(&targets)?),
        })
    }
}

#[derive(Default)]
pub struct UnsupervisedStackCollator;
impl UnsupervisedStackCollator {
    pub fn new() -> Self {
        Self
    }
}
impl Collator<UnsupervisedSample> for UnsupervisedStackCollator {
    type Batch = UnsupervisedBatch;
    fn collate(&mut self, samples: &[&UnsupervisedSample]) -> MlResult<Self::Batch> {
        let tensors = samples
            .iter()
            .map(|sample| &sample.input)
            .collect::<Vec<_>>();
        Ok(UnsupervisedBatch {
            samples: Variable::new(stack_tensors(&tensors)?),
        })
    }
}

/// Explicit same-shape autoregressive collator. It is never selected implicitly.
#[derive(Default)]
pub struct AutoregressiveStackCollator;
impl AutoregressiveStackCollator {
    pub fn new() -> Self {
        Self
    }
}
impl Collator<AutoregressiveSample> for AutoregressiveStackCollator {
    type Batch = AutoregressiveBatch;
    fn collate(&mut self, samples: &[&AutoregressiveSample]) -> MlResult<Self::Batch> {
        let tensors = samples
            .iter()
            .map(|sample| &sample.sequence)
            .collect::<Vec<_>>();
        Ok(AutoregressiveBatch {
            sequences: Variable::new(stack_tensors(&tensors)?),
        })
    }
}

pub struct SemiSupervisedDataLoader<LD, UD, LC, UC> {
    labeled: LD,
    unlabeled: UD,
    labeled_collator: LC,
    unlabeled_collator: UC,
    labeled_batch_size: usize,
    unlabeled_batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    labeled_indices: Vec<usize>,
    unlabeled_indices: Vec<usize>,
    cursor: usize,
}

pub struct SemiSupervisedDataLoaderBuilder<LD, UD, LC = MissingCollator, UC = MissingCollator> {
    labeled: LD,
    unlabeled: UD,
    labeled_collator: LC,
    unlabeled_collator: UC,
    labeled_batch_size: usize,
    unlabeled_batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<LD, UD> SemiSupervisedDataLoader<LD, UD, MissingCollator, MissingCollator> {
    pub fn builder(labeled: LD, unlabeled: UD) -> SemiSupervisedDataLoaderBuilder<LD, UD> {
        SemiSupervisedDataLoaderBuilder {
            labeled,
            unlabeled,
            labeled_collator: MissingCollator,
            unlabeled_collator: MissingCollator,
            labeled_batch_size: 1,
            unlabeled_batch_size: 1,
            shuffle: true,
            drop_last: false,
        }
    }
}

impl<LD, UD, LC, UC> SemiSupervisedDataLoaderBuilder<LD, UD, LC, UC> {
    pub fn labeled_collator<NC>(
        self,
        collator: NC,
    ) -> SemiSupervisedDataLoaderBuilder<LD, UD, NC, UC> {
        SemiSupervisedDataLoaderBuilder {
            labeled_collator: collator,
            labeled: self.labeled,
            unlabeled: self.unlabeled,
            unlabeled_collator: self.unlabeled_collator,
            labeled_batch_size: self.labeled_batch_size,
            unlabeled_batch_size: self.unlabeled_batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
        }
    }
    pub fn unlabeled_collator<NC>(
        self,
        collator: NC,
    ) -> SemiSupervisedDataLoaderBuilder<LD, UD, LC, NC> {
        SemiSupervisedDataLoaderBuilder {
            unlabeled_collator: collator,
            labeled: self.labeled,
            unlabeled: self.unlabeled,
            labeled_collator: self.labeled_collator,
            labeled_batch_size: self.labeled_batch_size,
            unlabeled_batch_size: self.unlabeled_batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
        }
    }
    pub fn labeled_batch_size(mut self, value: usize) -> Self {
        self.labeled_batch_size = value;
        self
    }
    pub fn unlabeled_batch_size(mut self, value: usize) -> Self {
        self.unlabeled_batch_size = value;
        self
    }
    pub fn shuffle(mut self, value: bool) -> Self {
        self.shuffle = value;
        self
    }
    pub fn drop_last(mut self, value: bool) -> Self {
        self.drop_last = value;
        self
    }
}

impl<LD, UD, LC, UC> SemiSupervisedDataLoaderBuilder<LD, UD, LC, UC>
where
    LD: Dataset<Sample = SupervisedSample>,
    UD: Dataset<Sample = UnsupervisedSample>,
    LC: Collator<SupervisedSample, Batch = SupervisedBatch>,
    UC: Collator<UnsupervisedSample, Batch = UnsupervisedBatch>,
{
    pub fn build(self) -> MlResult<SemiSupervisedDataLoader<LD, UD, LC, UC>> {
        if self.labeled.is_empty() || self.unlabeled.is_empty() {
            return Err(DataError::EmptyDataset.into());
        }
        if self.labeled_batch_size == 0 || self.unlabeled_batch_size == 0 {
            return Err(DataError::InvalidBatchSize.into());
        }
        let labeled_count =
            batch_count(self.labeled.len(), self.labeled_batch_size, self.drop_last);
        let unlabeled_count = batch_count(
            self.unlabeled.len(),
            self.unlabeled_batch_size,
            self.drop_last,
        );
        if labeled_count == 0 || unlabeled_count == 0 {
            return Err(DataError::NoBatches.into());
        }
        Ok(SemiSupervisedDataLoader {
            labeled_indices: (0..self.labeled.len()).collect(),
            unlabeled_indices: (0..self.unlabeled.len()).collect(),
            labeled: self.labeled,
            unlabeled: self.unlabeled,
            labeled_collator: self.labeled_collator,
            unlabeled_collator: self.unlabeled_collator,
            labeled_batch_size: self.labeled_batch_size,
            unlabeled_batch_size: self.unlabeled_batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            cursor: 0,
        })
    }
}

fn batch_count(len: usize, size: usize, drop_last: bool) -> usize {
    if drop_last {
        len / size
    } else {
        len.div_ceil(size)
    }
}

fn batch_indices(indices: &[usize], size: usize, batch: usize, drop_last: bool) -> &[usize] {
    let count = batch_count(indices.len(), size, drop_last);
    let logical = batch % count;
    let start = logical * size;
    let end = (start + size).min(indices.len());
    &indices[start..end]
}

impl<LD, UD, LC, UC> BatchLoader for SemiSupervisedDataLoader<LD, UD, LC, UC>
where
    LD: Dataset<Sample = SupervisedSample>,
    UD: Dataset<Sample = UnsupervisedSample>,
    LC: Collator<SupervisedSample, Batch = SupervisedBatch>,
    UC: Collator<UnsupervisedSample, Batch = UnsupervisedBatch>,
{
    type Batch = SemiSupervisedBatch;

    fn begin_epoch(&mut self, _epoch: usize, runtime: &TrainingRuntime) -> MlResult<()> {
        self.labeled_indices.clear();
        self.labeled_indices.extend(0..self.labeled.len());
        self.unlabeled_indices.clear();
        self.unlabeled_indices.extend(0..self.unlabeled.len());
        if self.shuffle {
            runtime.shuffle(&mut self.labeled_indices);
            runtime.shuffle(&mut self.unlabeled_indices);
        }
        self.cursor = 0;
        Ok(())
    }

    fn next_batch(&mut self) -> MlResult<Option<Self::Batch>> {
        let total = self.batch_count().unwrap_or(0);
        if self.cursor >= total {
            return Ok(None);
        }
        let labeled_indices = batch_indices(
            &self.labeled_indices,
            self.labeled_batch_size,
            self.cursor,
            self.drop_last,
        );
        let unlabeled_indices = batch_indices(
            &self.unlabeled_indices,
            self.unlabeled_batch_size,
            self.cursor,
            self.drop_last,
        );
        let labeled_samples = labeled_indices
            .iter()
            .map(|&i| self.labeled.get(i).unwrap())
            .collect::<Vec<_>>();
        let unlabeled_samples = unlabeled_indices
            .iter()
            .map(|&i| self.unlabeled.get(i).unwrap())
            .collect::<Vec<_>>();
        self.cursor += 1;
        let labeled = self
            .labeled_collator
            .collate(&labeled_samples)
            .map_err(classify_collate_error)?;
        let unlabeled = self
            .unlabeled_collator
            .collate(&unlabeled_samples)
            .map_err(classify_collate_error)?;
        Ok(Some(SemiSupervisedBatch {
            labeled_inputs: labeled.inputs,
            labeled_targets: labeled.targets,
            unlabeled_inputs: unlabeled.samples,
        }))
    }

    fn batch_count(&self) -> Option<usize> {
        Some(
            batch_count(self.labeled.len(), self.labeled_batch_size, self.drop_last).max(
                batch_count(
                    self.unlabeled.len(),
                    self.unlabeled_batch_size,
                    self.drop_last,
                ),
            ),
        )
    }
}
