use super::*;

/// 테스트를 위한 로거를 설정합니다.
/// 콘솔과 파일에 동시에 로그를 남깁니다.
pub fn setup_logger() {
    use tracing_subscriber::{prelude::*, EnvFilter, fmt};

    let file_appender = tracing_appender::rolling::minutely("logs", "test_run.log");
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));

    let file_layer = fmt::layer().with_writer(non_blocking_appender).with_ansi(false);
    let stdout_layer = fmt::layer().with_writer(std::io::stdout);

    // 다른 테스트와 충돌하지 않도록 try_init 사용
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(stdout_layer)
        .try_init();
    // _guard는 의도적으로 drop시킴
}

/// 모델의 학습된 파라미터를 JSON 파일로 저장합니다.
pub fn save_model_parameters(mlp: &MLP, path: &str) {
    info!("Saving model parameters to '{}'...", path);
    let params_to_save = ModelParameters {
        w1_data: mlp.w1.tensor().data().to_vec(),
        w1_shape: mlp.w1.tensor().shape().to_vec(),
        b1_data: mlp.b1.tensor().data().to_vec(),
        b1_shape: mlp.b1.tensor().shape().to_vec(),
        w2_data: mlp.w2.tensor().data().to_vec(),
        w2_shape: mlp.w2.tensor().shape().to_vec(),
        b2_data: mlp.b2.tensor().data().to_vec(),
        b2_shape: mlp.b2.tensor().shape().to_vec(),
    };

    match File::create(path) {
        Ok(file) => {
            if let Err(e) = serde_json::to_writer_pretty(file, &params_to_save) {
                warn!("Failed to save model to JSON: {}", e);
            } else {
                info!("Successfully saved model parameters.");
            }
        }
        Err(e) => warn!("Failed to create file '{}': {}", path, e),
    }
}

/// 계산 그래프를 SVG 파일로 렌더링합니다. (`enableVisualization` 피처가 활성화된 경우에만 동작)
pub fn generate_visualization(path: &str) {
    #[cfg(feature = "enableVisualization")]
    {
        info!("Generating computation graph visualization...");
        match crate::tensor::VisualizationGraph::render_to_svg(path) {
            Ok(_) => info!("SVG graph saved to '{}'", path),
            Err(e) => warn!("Failed to save SVG graph: {:?}", e),
        }
    }
    #[cfg(not(feature = "enableVisualization"))]
    {
        // 피처가 비활성화된 경우 아무 작업도 하지 않도록 명시
        let _ = path;
    }
}
