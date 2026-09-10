use super::*;
use crate::legacy::{nn::Variable, tensor::{Tensor, TensorBase}};
#[test] fn schedules_reject_zero() {
    assert!(EpochSchedule::new(0).is_err());
    assert!(EpisodeSchedule::new(1, 0).is_err());
}

#[test] fn datasets_validate_contracts() {
    let x = Variable::new(Tensor::from_vec(vec![1.0], &[1, 1]).unwrap());
    let t = Variable::new(Tensor::from_vec(vec![0.0], &[1, 1]).unwrap());
    let xs = [&x]; let ts = [&t];
    assert!(SupervisedDataset::new(&xs, &ts).is_ok());
    assert!(SupervisedDataset::new(&[], &[]).is_err());
    assert!(SupervisedDataset::new(&xs, &[]).is_err());
    assert!(UnsupervisedDataset::new(&[]).is_err());
    assert!(AutoregressiveDataset::new(&[]).is_err());
    assert!(SemiSupervisedDataset::new(&xs, &ts, &[]).is_err());
}

#[test] fn identical_seed_produces_identical_stream() {
    let a = super::super::TrainingRuntime::new(7);
    let b = super::super::TrainingRuntime::new(7);
    let mut left = vec![1, 2, 3, 4, 5];
    let mut right = left.clone();
    a.shuffle(&mut left); b.shuffle(&mut right);
    assert_eq!(left, right);
    assert_eq!(a.random_f32(), b.random_f32());
}

#[test] fn facade_selectors_are_statically_typed() {
    let _: super::super::SupervisedTrainer = super::super::Trainer::silent().supervised();
    let _: super::super::UnsupervisedTrainer = super::super::Trainer::silent().unsupervised();
    let _: super::super::SemiSupervisedTrainer = super::super::Trainer::silent().semi_supervised();
    let _: super::super::AutoregressiveTrainer = super::super::Trainer::silent().autoregressive();
    let _: super::super::RLTrainer = super::super::Trainer::silent().reinforcement();
}
