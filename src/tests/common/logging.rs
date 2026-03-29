use super::*;

pub fn setup_logging() -> tracing_appender::non_blocking::WorkerGuard {
    let file_appender = tracing_appender::rolling::hourly("logs", "test_run.log");
    let (non_blocking_appender, guard) = tracing_appender::non_blocking(file_appender);
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("debug"));
    let time_format = format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]");
    let file_layer = fmt::layer()
        .with_writer(non_blocking_appender)
        .with_timer(fmt::time::UtcTime::new(time_format))
        .with_ansi(false);
    let stdout_layer = fmt::layer()
        .with_writer(std::io::stdout)
        .with_timer(fmt::time::UtcTime::new(time_format));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(stdout_layer)
        .try_init();

    guard
}
