use trench_deep::{
    MlResult, TensorBuffer,
    trainer::{ClassificationAccuracy, argmax},
};

#[test]
fn accuracy_counts_each_sample_in_a_batch() -> MlResult<()> {
    let prediction = TensorBuffer::from_vec(vec![9.0, 1.0, 2.0, 3.0, 5.0, 1.0], &[3, 2])?;
    let target = TensorBuffer::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[3, 2])?;
    let mut metric = ClassificationAccuracy::new();
    metric.update(&prediction, &target);
    assert!((metric.compute() - 200.0 / 3.0).abs() < 1e-5);
    metric.reset();
    assert_eq!(metric.compute(), 0.0);
    Ok(())
}

#[test]
fn argmax_preserves_first_tie() {
    assert_eq!(argmax(&[2.0, 2.0, 1.0]), Some(0));
    assert_eq!(argmax(&[]), None);
}
