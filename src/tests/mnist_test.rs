use super::*;

use crate::trainer::{ClassificationAccuracy, Trainer, SupervisedDataset, EpochSchedule};

/// 테스트용 간이 평가기: argmax 정확도를 누적 계산해 백분율 반환.
#[cfg(feature = "enableBackward")]
fn evaluate_accuracy<M: Model>(model: &mut M, x: &[&Variable], t: &[&Variable]) -> MlResult<f32> {
    let mut acc = ClassificationAccuracy::new();
    for (xi, ti) in x.iter().zip(t.iter()) {
        let y = model.predict(xi.tensor())?;
        acc.update(&y, ti.tensor());
    }
    Ok(acc.compute())
}

#[test]
#[cfg(all(feature = "enableBackward"))]
fn mlp_mnist_classification_integration_test() -> MlResult<()> {
    let _guard = setup_logging();
    let config = MnistConfig::default();
    info!("=== Starting MLP MNIST Classification Test ===");
    info!("{:?}", config);

    let dataset = MnistDataset::load_and_prepare_data(config.n_train, config.n_val, config.n_features, config.n_classes)?;
    let mut mlp = MLP::build_model(config.n_features, config.n_hidden_2, config.n_classes)?;
    let mut opt = SGD::new(config.learning_rate);
    for p in crate::trainer::TrainableModel::params(&mlp) {
        opt.register(p);
    }

    let x_train = dataset.x_train();
    let t_train = dataset.t_train();
    Trainer::default().supervised().fit(&mut mlp, &mut opt,
        SupervisedDataset::new(&x_train, &t_train)?,
        EpochSchedule::new(config.epochs)?.with_tolerance(config.tolerance))?;

    let accuracy = evaluate_accuracy(&mut mlp, &dataset.x_train(), &dataset.t_train())?;

    if accuracy > config.required_accuracy {
        info!("Target accuracy achieved! ({:.2}% > {:.2}%)", accuracy, config.required_accuracy);
        crate::trainer::CheckpointableModel::save_checkpoint(&mlp, std::path::Path::new(&config.model_save_path))?;
    } else {
        warn!("Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).", accuracy, config.required_accuracy);
    }

    generate_visualization(&config.visualization_path);
    info!("=== MLP MNIST Test Finished ===");
    assert!(accuracy > config.required_accuracy, "Model did not reach the required accuracy threshold.");

    Ok(())
}

#[test]
#[ignore]
#[cfg(all(feature = "enableBackward"))]
fn softmax_regression_mnist_classification_integration_test() -> MlResult<()> {
    let _guard = setup_logging();
    let config = MnistConfig::default();
    info!("=== Starting SoftmaxRegression MNIST Classification Test ===");
    info!("{:?}", config);

    let dataset = MnistDataset::load_and_prepare_data(config.n_train, config.n_val, config.n_features, config.n_classes)?;
    let mut model = SoftmaxRegression::build_model(config.n_features, config.n_classes)?;

    info!("Starting model training...");
    info!("Training Parameters: LR={}, Max Epochs={}, Tolerance={}", config.learning_rate, config.epochs, config.tolerance);

    let mut opt = SGD::new(config.learning_rate);
    opt.register(&model.w1);
    opt.register(&model.b1);
    let x_train = dataset.x_train();
    let t_train = dataset.t_train();
    Trainer::default().supervised().fit(&mut model, &mut opt,
        SupervisedDataset::new(&x_train, &t_train)?,
        EpochSchedule::new(config.epochs)?.with_tolerance(config.tolerance))?;

    info!("Model training finished.");

    let accuracy = evaluate_accuracy(&mut model, &dataset.x_train(), &dataset.t_train())?;
    if accuracy > config.required_accuracy {
        info!("Target accuracy achieved! ({:.2}% > {:.2}%)", accuracy, config.required_accuracy);
        crate::trainer::CheckpointableModel::save_checkpoint(&model, std::path::Path::new(&config.model_save_path))?;
    } else {
        warn!("Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).", accuracy, config.required_accuracy);
    }

    generate_visualization(&config.visualization_path);
    info!("=== SoftmaxRegression MNIST Test Finished ===");
    assert!(accuracy > config.required_accuracy, "Model did not reach the required accuracy threshold.");
    Ok(())
}
