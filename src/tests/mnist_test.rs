use super::*;

#[ignore]
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

    mlp.train(
        &dataset.x_train(),
        &dataset.t_train(),
        config.epochs,
        &mut opt,
        config.tolerance,
    )?;

    let accuracy = mlp.evaluate_model(&dataset.x_train(), &dataset.t_train())?;

    if accuracy > config.required_accuracy {
        info!("Target accuracy achieved! ({:.2}% > {:.2}%)", accuracy, config.required_accuracy);
        mlp.save(&config.model_save_path)?;
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

    #[cfg(feature = "enableBackward")]
    {
        let mut opt = SGD::new(config.learning_rate);
        opt.register(&model.w1);
        opt.register(&model.b1);
        model.train(&dataset.x_train(), &dataset.t_train(), config.epochs, &mut opt, config.tolerance)?;
    }
    if !cfg!(feature = "enableBackward") {
        warn!("Feature: disableBackpropagation");
    }

    info!("Model training finished.");

    let accuracy = model.evaluate_model(&dataset.x_train(), &dataset.t_train())?;
    if accuracy > config.required_accuracy {
        info!("Target accuracy achieved! ({:.2}% > {:.2}%)", accuracy, config.required_accuracy);
        model.save(&config.model_save_path)?;
    } else {
        warn!("Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).", accuracy, config.required_accuracy);
    }

    generate_visualization(&config.visualization_path);
    info!("=== SoftmaxRegression MNIST Test Finished ===");
    assert!(accuracy > config.required_accuracy, "Model did not reach the required accuracy threshold.");
    Ok(())
}
