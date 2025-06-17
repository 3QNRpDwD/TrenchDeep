//! tests/mnist_test
//! MLP 모델의 MNIST 분류 성능을 검증하는 통합 테스트입니다.

// `super::*`를 통해 `src` 라이브러리의 아이템들을 가져옵니다.
use super::*;
use crate::MlResult;
use log::{info, warn};

// 새로 만든 공용 모듈들을 임포트합니다.
use common::config::TestConfig;
use common::data::load_and_prepare_data;
use common::evaluation::evaluate_model;
use common::utils::{generate_visualization, setup_logging};
use crate::tests::common::{Model, SoftmaxRegression, MLP};

#[test]
fn mlp_mnist_classification_integration_test() -> MlResult<()> {
    // 1. 설정 (Setup)
    // 로거와 테스트 설정을 초기화합니다.
    let a = setup_logging();
    let config = TestConfig::default();
    info!("=== Starting MLP MNIST Classification Test with Config ===");
    info!("{:?}", config);

    let dataset =
        load_and_prepare_data(config.n_train, config.n_val, config.n_features, config.n_classes)?;
    
    let mut mlp = MLP::build_model(config.n_features, config.n_hidden_2, config.n_classes)?;
    MLP::train_model(
        &mut mlp,
        &dataset.x_train,
        &dataset.t_train,
        config.learning_rate,
        config.epochs,
        config.tolerance,
    )?;
    
    let accuracy = evaluate_model(&mlp, &dataset.x_test, &dataset.t_test)?;

    if accuracy > config.required_accuracy {
        info!(
            "🎉 Target accuracy achieved! ({:.2}% > {:.2}%)",
            accuracy, config.required_accuracy
        );
        // 성공 시 모델 파라미터를 저장합니다.
        mlp.save(&config.model_save_path)?;
    } else {
        warn!(
            "⚠️ Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).",
            accuracy, config.required_accuracy
        );
    }

    // (선택) 계산 그래프를 시각화합니다.
    generate_visualization(&config.visualization_path);
    info!("=== MLP MNIST Test Finished ===");
    assert!(
        accuracy > config.required_accuracy,
        "Model did not reach the required accuracy threshold."
    );

    Ok(())
}

#[test]
fn softmax_regression_mnist_classification_integration_test() -> MlResult<()> {
    // 1. 설정 (Setup)
    // 로거와 테스트 설정을 초기화합니다.
    let a = setup_logging();
    let config = TestConfig::default();
    info!("=== Starting MLP MNIST Classification Test with Config ===");
    info!("{:?}", config);

    let dataset =
        load_and_prepare_data(config.n_train, config.n_val, config.n_features, config.n_classes)?;

    let mut model = SoftmaxRegression::build_model(config.n_features, config.n_classes)?;
    SoftmaxRegression::train_model(
        &mut model,
        &dataset.x_train,
        &dataset.t_train,
        config.learning_rate,
        config.epochs,
        config.tolerance,
    )?;

    let accuracy = evaluate_model(&model, &dataset.x_test, &dataset.t_test)?;

    if accuracy > config.required_accuracy {
        info!(
            "🎉 Target accuracy achieved! ({:.2}% > {:.2}%)",
            accuracy, config.required_accuracy
        );
        // 성공 시 모델 파라미터를 저장합니다.
        model.save(&config.model_save_path)?;
    } else {
        warn!(
            "⚠️ Target accuracy NOT met. (Actual: {:.2}%, Required: {:.2}%).",
            accuracy, config.required_accuracy
        );
    }

    // (선택) 계산 그래프를 시각화합니다.
    generate_visualization(&config.visualization_path);
    info!("=== MLP MNIST Test Finished ===");
    assert!(
        accuracy > config.required_accuracy,
        "Model did not reach the required accuracy threshold."
    );

    Ok(())
}
