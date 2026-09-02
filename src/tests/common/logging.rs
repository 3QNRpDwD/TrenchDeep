use super::*;
use std::sync::OnceLock;

static LOGGING_GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

/// 테스트 프로세스의 전역 tracing subscriber를 한 번만 초기화한다.
///
/// `WorkerGuard`는 non-blocking writer보다 오래 살아야 하므로 정적 저장소에
/// 보관한다. 테스트별 지역 변수로 반환하면 최초 테스트 종료 후 writer가
/// 닫히고, 이후 테스트의 로그가 유실될 수 있다.
pub fn setup_logging() -> &'static tracing_appender::non_blocking::WorkerGuard {
    LOGGING_GUARD.get_or_init(|| {
        let file_appender = tracing_appender::rolling::hourly("logs", "test_run.log");
        let (non_blocking_appender, guard) = tracing_appender::non_blocking(file_appender);
        // 애플리케이션 실행용 RUST_LOG가 테스트 로그를 우연히 억제하지 않도록
        // 테스트 전용 환경 변수를 사용한다. 예: TEST_LOG=trace cargo test ...
        let filter = EnvFilter::try_from_env("TEST_LOG").unwrap_or_else(|_| {
            if cfg!(feature = "debugging") {
                EnvFilter::new("trench_deep=trace")
            } else {
                EnvFilter::new("debug")
            }
        });
        let time_format = format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]");
        let file_layer = fmt::layer()
            .with_writer(non_blocking_appender)
            .with_timer(fmt::time::UtcTime::new(time_format))
            .with_ansi(false);
        let stdout_layer = fmt::layer()
            .with_writer(std::io::stdout)
            .with_timer(fmt::time::UtcTime::new(time_format));

        tracing_subscriber::registry()
            .with(filter)
            .with(file_layer)
            .with(stdout_layer)
            .try_init()
            .expect("test logging subscriber must be initialized exactly once");

        guard
    })
}
