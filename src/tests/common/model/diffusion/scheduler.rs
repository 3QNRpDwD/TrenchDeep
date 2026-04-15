use super::*;

trait DiffusionScheduler {

}

pub struct Scheduler {
    pub scheduler: Box<dyn DiffusionScheduler>,
}