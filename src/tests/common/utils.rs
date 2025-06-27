use super::*;

pub fn setup_logging() -> tracing_appender::non_blocking::WorkerGuard {
    let file_appender = tracing_appender::rolling::hourly("logs", "test_run.log");
    let (non_blocking_appender, guard) = tracing_appender::non_blocking(file_appender);
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("debug"));
    let time_format = format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]");
    let file_layer = fmt::layer()
        .with_writer(non_blocking_appender)
        .with_timer(tracing_subscriber::fmt::time::UtcTime::new(time_format))
        .with_ansi(false);
    let stdout_layer = fmt::layer()
        .with_writer(std::io::stdout)
        .with_timer(tracing_subscriber::fmt::time::UtcTime::new(time_format));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(stdout_layer)
        .try_init();

    guard
}

pub fn argmax(data: &[f32]) -> Option<usize> {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}



impl MLP {
    pub fn build_model(n_input: usize, n_hidden : usize, n_output: usize) -> MlResult<MLP> {
        let hidden_activation = Sigmoid::new()?;
        let output_activation = Softmax::new()?;
        let loss_function = CrossEntropyLoss::new()?;
        let layer = Sequential::new();

        info!("Network Structure: {}(Input) -> {}(Hidden) -> {}(Output)", n_input, n_hidden, n_output);
        info!("Activation Functions: {} (Hidden), {} (Output)", hidden_activation.name(), output_activation.name());

        let mlp = MLP::new(&[n_input, n_hidden, n_output], layer, loss_function);
        info!("MLP model created successfully.");
        Ok(mlp)
    }

    //모델의 학습이 더이상 진행되지 않는 상황에서 파라미터를 조정해봤으나, 유의미한 영향이 있지않았음.
    // 오히려 학습률이 비정상적으로 작아지는등 모습을 보임.
    // 따라서 레이어를 하나 더 추가했으나,이도 유의미한 결과를 내지 못하고있는것으로 보임. 마지막 방법으로, 옵티마이저를 적응형으로 변경하는 방안을 고려. 그 이후에도 해결되지 않는다면...
}

impl SoftmaxRegression {
    pub fn build_model(n_input: usize, n_output: usize) -> MlResult<SoftmaxRegression> {
        let loss_function = SoftmaxCrossEntropyLoss::new()?;
        info!("Network Structure: {}(Input) -> {}(Output)", n_input, n_output);
        let sr = SoftmaxRegression::new(&[n_input, n_output], loss_function);
        info!("MLP model created successfully.");
        Ok(sr)
    }

    //모델의 학습이 더이상 진행되지 않는 상황에서 파라미터를 조정해봤으나, 유의미한 영향이 있지않았음.
    // 오히려 학습률이 비정상적으로 작아지는등 모습을 보임.
    // 따라서 레이어를 하나 더 추가했으나,이도 유의미한 결과를 내지 못하고있는것으로 보임. 마지막 방법으로, 옵티마이저를 적응형으로 변경하는 방안을 고려. 그 이후에도 해결되지 않는다면...

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
