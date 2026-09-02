use std::cell::RefCell;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

/// Deterministic single-threaded runtime shared by every trainer paradigm.
/// TODO(ParallelRuntime): add a Send + Sync implementation with parallel training.
pub struct TrainingRuntime { rng: RefCell<StdRng> }

impl TrainingRuntime {
    pub fn new(seed: u64) -> Self { Self { rng: RefCell::new(StdRng::seed_from_u64(seed)) } }
    pub fn reseed(&self, seed: u64) { *self.rng.borrow_mut() = StdRng::seed_from_u64(seed); }
    pub fn shuffle<T>(&self, values: &mut [T]) { values.shuffle(&mut *self.rng.borrow_mut()); }
    pub fn random_f32(&self) -> f32 { self.rng.borrow_mut().random::<f32>() }
}
