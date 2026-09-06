use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::cell::RefCell;

/// Deterministic single-threaded runtime shared by every trainer paradigm.
/// TODO(ParallelRuntime): add a Send + Sync implementation with parallel training.
pub struct TrainingRuntime {
    data_rng: RefCell<StdRng>,
    action_rng: RefCell<StdRng>,
}

impl TrainingRuntime {
    pub fn new(seed: u64) -> Self {
        Self {
            data_rng: RefCell::new(StdRng::seed_from_u64(seed)),
            action_rng: RefCell::new(StdRng::seed_from_u64(seed ^ 0xA076_1D64_78BD_642F)),
        }
    }
    pub fn reseed(&self, seed: u64) {
        *self.data_rng.borrow_mut() = StdRng::seed_from_u64(seed);
        *self.action_rng.borrow_mut() = StdRng::seed_from_u64(seed ^ 0xA076_1D64_78BD_642F);
    }
    pub fn shuffle<T>(&self, values: &mut [T]) {
        values.shuffle(&mut *self.data_rng.borrow_mut());
    }
    /// RL action randomness is independent of data shuffling.
    pub fn random_f32(&self) -> f32 {
        self.action_rng.borrow_mut().random::<f32>()
    }
}
