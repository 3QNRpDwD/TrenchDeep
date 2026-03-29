use super::*;

// checkpoint 전용 import
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

// ────────────────────────────────────────────────────────────────────────────
// 시그널 핸들러 — 프로세스당 한 번만 등록
// ────────────────────────────────────────────────────────────────────────────

static INTERRUPT_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Ctrl+C 인터럽트 플래그를 반환한다.
/// 최초 호출 시 시그널 핸들러를 등록하며, 이후 호출에서는 같은 플래그를 공유한다.
pub(crate) fn interrupt_flag() -> Arc<AtomicBool> {
    INTERRUPT_FLAG.get_or_init(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();
        ctrlc::set_handler(move || {
            flag_clone.store(true, Ordering::SeqCst);
        }).expect("Ctrl+C 핸들러 등록 실패");
        flag
    }).clone()
}

/// 인터럽트 플래그가 설정되어 있는지 확인한다.
pub(crate) fn is_interrupted(flag: &AtomicBool) -> bool {
    flag.load(Ordering::SeqCst)
}

/// 인터럽트 플래그를 초기화한다.
pub(crate) fn clear_interrupt(flag: &AtomicBool) {
    flag.store(false, Ordering::SeqCst);
}

// ────────────────────────────────────────────────────────────────────────────
// 사용자 확인 프롬프트
// ────────────────────────────────────────────────────────────────────────────

/// 사용자에게 학습 중단 여부를 확인한다.
///
/// `true`를 반환하면 체크포인트를 저장하고 종료,
/// `false`를 반환하면 학습을 계속한다.
pub(crate) fn confirm_interrupt() -> bool {
    use std::io::{self, BufRead, Write};

    eprint!("\n⚠ 학습 중단 요청이 감지되었습니다.\n");
    eprint!("체크포인트를 저장하고 종료하시겠습니까? (y/n): ");
    io::stderr().flush().ok();

    let mut input = String::new();
    match io::stdin().lock().read_line(&mut input) {
        Ok(_) => input.trim().eq_ignore_ascii_case("y"),
        Err(_) => {
            // stdin 읽기 실패 시 안전하게 저장 후 종료
            eprintln!("입력을 읽을 수 없습니다. 안전하게 체크포인트를 저장합니다.");
            true
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TrainingCheckpoint — 학습 상태 직렬화
// ────────────────────────────────────────────────────────────────────────────

/// 학습 중단 시 저장되는 체크포인트.
///
/// 에폭 단위로 저장되며, 재개 시 다음 에폭부터 학습을 이어간다.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingCheckpoint {
    /// 완료한 에폭 수 (재개 시 이 값부터 시작)
    pub epochs_done: usize,
    /// 목표 총 에폭 수
    pub total_epochs: usize,
    /// 마지막 에폭의 평균 손실
    pub last_loss: f32,
    /// 수렴 판정 tolerance
    pub tolerance: f32,
    /// 옵티마이저 학습률
    pub optimizer_lr: f32,
    /// 모델 가중치 파일 경로
    pub model_path: String,
    /// 체크포인트 저장 시각
    pub timestamp: String,
}

impl TrainingCheckpoint {
    /// 체크포인트를 JSON 파일로 저장한다.
    pub fn save(&self, path: &str) -> MlResult<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| MlError::StringError(
                        format!("디렉토리 생성 실패 '{}': {}", parent.display(), e)
                    ))?;
            }
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| MlError::StringError(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| MlError::StringError(
                format!("체크포인트 저장 실패 '{}': {}", path, e)
            ))
    }

    /// JSON 파일에서 체크포인트를 로드.
    pub fn load(path: &str) -> MlResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| MlError::StringError(
                format!("체크포인트 로드 실패 '{}': {}", path, e)
            ))?;
        serde_json::from_str(&json)
            .map_err(|e| MlError::StringError(
                format!("체크포인트 파싱 실패 '{}': {}", path, e)
            ))
    }

    /// 체크포인트 파일이 존재하는지 확인.
    pub fn exists(path: &str) -> bool {
        std::path::Path::new(path).exists()
    }
}
