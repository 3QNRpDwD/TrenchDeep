use once_cell::sync::Lazy;
use std::sync::Mutex;
use num_cpus::get;
#[cfg(feature = "parallel")]
use rayon::{ThreadPool, ThreadPoolBuilder};

/// 병렬 처리 설정을 담는 구조체
pub struct ParallelConfig {
    #[cfg(feature = "parallel")]
    pub(crate) pool: Option<ThreadPool>,
    pub num_threads: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let num_threads = get();
        Self {
            #[cfg(feature = "parallel")]
            pool: Some(ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap()),
            num_threads,
        }
    }
}

/// 전역으로 관리되는 병렬 처리 설정
pub static PARALLEL_CONFIG: Lazy<Mutex<ParallelConfig>> = Lazy::new(|| Mutex::new(ParallelConfig::default()));

/// 사용할 스레드 수를 설정합니다.
///
/// # Arguments
/// * `num_threads`: 사용할 스레드 수. 0으로 설정 시 사용 가능한 모든 코어를 사용합니다.
#[cfg(feature = "parallel")]
pub fn set_num_threads(num_threads: usize) {
    let mut config = PARALLEL_CONFIG.lock().unwrap();
    let new_num_threads = if num_threads == 0 { get() } else { num_threads };

    config.num_threads = new_num_threads;
    config.pool = Some(
        ThreadPoolBuilder::new()
            .num_threads(new_num_threads)
            .build()
            .unwrap(),
    );
}

/// 설정 함수 (parallel feature 비활성화 시)
#[cfg(not(feature = "parallel"))]
pub fn set_num_threads(_num_threads: usize) {
    // 병렬 기능이 비활성화되었으므로 아무 작업도 하지 않음
    println!("Warning: 'parallel' feature is not enabled. set_num_threads() has no effect.");
}