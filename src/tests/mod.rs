pub mod common;

#[cfg(test)]
mod mnist_test {
    use super::*;
    use log::{info, warn};

    use crate::MlResult;
    use crate::tests::common::config::TestConfig;
    use crate::tests::common::data::load_and_prepare_data;
    use crate::tests::common::evaluation::evaluate_model;
    use crate::tests::common::utils::{generate_visualization, setup_logging};
    use crate::tensor::TENSOR_STORAGE;
    use crate::tests::common::model::{Model, SoftmaxRegression, MLP};

    #[test]
    fn mlp_mnist_classification_integration_test() -> MlResult<()> {
        let _ = setup_logging();
        let config = TestConfig::default();
        info!("=== Starting MLP MNIST Classification Test with Config ===");
        info!("{:?}", config);

        let dataset = load_and_prepare_data(config.n_train, config.n_val, config.n_features, config.n_classes)?;
        let mut mlp = MLP::build_model(config.n_features, config.n_hidden_2, config.n_classes)?;

        MLP::train_model(
            &mut mlp,
            &dataset.x_train,
            &dataset.t_train,
            config.learning_rate,
            config.epochs,
            config.tolerance,
        )?;

        let accuracy = evaluate_model(&mut mlp, &dataset.x_test, &dataset.t_test)?;

        if accuracy > config.required_accuracy {
            info!("🎉 Target accuracy achieved! ({:.2}% > {:.2}%)",accuracy, config.required_accuracy);
            mlp.save(&config.model_save_path)?;
        } else {
            warn!("⚠️ Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).",accuracy, config.required_accuracy);
        }

        // (선택) 계산 그래프를 시각화합니다.
        generate_visualization(&config.visualization_path);
        info!("=== MLP MNIST Test Finished ===");
        assert!(accuracy > config.required_accuracy, "Model did not reach the required accuracy threshold.");

        Ok(())
    }

    #[test]
    fn softmax_regression_mnist_classification_integration_test() -> MlResult<()> {
        let _ = setup_logging();
        let config = TestConfig::default();
        info!("=== Starting MLP MNIST Classification Test with Config ===");
        info!("{:?}", config);

        let dataset = load_and_prepare_data(config.n_train, config.n_val, config.n_features, config.n_classes)?;
        let mut model = SoftmaxRegression::build_model(config.n_features, config.n_classes)?;

        info!("Starting model training...");
        info!("Training Parameters: LR={}, Max Epochs={}, Tolerance={}", config.learning_rate, config.epochs, config.tolerance);

        #[cfg(feature = "enableBackpropagation")]
        model.train(&dataset.x_train, &dataset.t_train, config.epochs, config.learning_rate, config.tolerance)?;
        if !cfg!(feature = "enableBackpropagation") {
            warn!("Feature: disableBackpropagation");
        }

        info!("Model training finished.");

        let accuracy = evaluate_model(&mut model, &dataset.x_test, &dataset.t_test)?;
        if accuracy > config.required_accuracy {
            info!("🎉 Target accuracy achieved! ({:.2}% > {:.2}%)",accuracy, config.required_accuracy);
            model.save(&config.model_save_path)?;
        } else {
            warn!("⚠️ Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).",accuracy, config.required_accuracy);
        }

        generate_visualization(&config.visualization_path);
        info!("=== MLP MNIST Test Finished ===");
        assert!(accuracy > config.required_accuracy, "Model did not reach the required accuracy threshold.");
        Ok(())
    }
}