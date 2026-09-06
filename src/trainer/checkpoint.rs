use super::*;
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 2;
fn default_schema_version() -> u32 {
    CHECKPOINT_SCHEMA_VERSION
}
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static HANDLER: OnceLock<Result<(), String>> = OnceLock::new();
pub fn request_interrupt() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}
pub fn clear_interrupt() {
    INTERRUPTED.store(false, Ordering::SeqCst);
}
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}
pub fn install_interrupt_handler() -> MlResult<()> {
    HANDLER
        .get_or_init(|| ctrlc::set_handler(request_interrupt).map_err(|e| e.to_string()))
        .clone()
        .map_err(MlError::StringError)
}
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParadigmTag {
    Supervised,
    Unsupervised,
    SemiSupervised,
    Autoregressive,
    Reinforcement,
}

impl ParadigmTag {
    pub fn as_str(self) -> &'static str {
        match self {
            ParadigmTag::Supervised => "supervised",
            ParadigmTag::Unsupervised => "unsupervised",
            ParadigmTag::SemiSupervised => "semi-supervised",
            ParadigmTag::Autoregressive => "autoregressive",
            ParadigmTag::Reinforcement => "reinforcement",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TrainingCheckpoint — 학습 상태 직렬화
// ────────────────────────────────────────────────────────────────────────────

/// 학습 중단 시 저장되는 체크포인트.
///
/// 에폭 단위로 저장되며, 재개 시 다음 에폭부터 학습을 이어간다.
///
/// ## 하위 호환성
///
/// Phase 4 에서 `paradigm` / `rng_seed` 필드가 추가되었다. 이전 버전에서
/// 저장된 JSON 은 두 필드가 누락되어 있으므로 `#[serde(default)]` 로 처리해
/// 무태그 체크포인트는 `None` / `0` 으로 로드된다. 로더가 패러다임 체크를
/// 강제하지 않는 한 구버전 체크포인트도 그대로 읽힌다.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingCheckpoint {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
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
    /// 이 체크포인트를 생성한 트레이너 패러다임. 구버전 호환을 위해 선택 필드.
    #[serde(default)]
    pub paradigm: Option<ParadigmTag>,
    /// 학습 시 사용한 RNG 시드. 재현성·셔플 결정성을 위해 기록. 0 이면 미기록.
    #[serde(default)]
    pub rng_seed: u64,
    /// TODO(Phase-6): populated once optimizer implementations support snapshots.
    #[serde(default)]
    pub optimizer_snapshot: Option<serde_json::Value>,
}

impl TrainingCheckpoint {
    /// 체크포인트를 JSON 파일로 저장한다.
    pub fn save(&self, path: impl AsRef<Path>) -> MlResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    MlError::StringError(format!(
                        "디렉토리 생성 실패 '{}': {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| MlError::StringError(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| {
            MlError::StringError(format!("체크포인트 저장 실패 '{}': {}", path.display(), e))
        })
    }

    /// JSON 파일에서 체크포인트를 로드.
    pub fn load(path: impl AsRef<Path>) -> MlResult<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|e| {
            MlError::StringError(format!("체크포인트 로드 실패 '{}': {}", path.display(), e))
        })?;
        serde_json::from_str(&json).map_err(|e| {
            MlError::StringError(format!("체크포인트 파싱 실패 '{}': {}", path.display(), e))
        })
    }

    /// 체크포인트 파일이 존재하는지 확인.
    pub fn exists(path: impl AsRef<Path>) -> bool {
        path.as_ref().exists()
    }

    /// `expected` 와 태그가 일치하는지 확인. 구버전(태그 없음) 체크포인트는
    /// 통과한다 (사용자가 명시적으로 패러다임을 선택한 것으로 간주).
    pub fn verify_paradigm(&self, expected: ParadigmTag) -> MlResult<()> {
        if let Some(tag) = self.paradigm {
            if tag != expected {
                return Err(MlError::StringError(format!(
                    "체크포인트 패러다임 불일치: 저장된 태그는 `{}`, \
                     현재 트레이너는 `{}`. 올바른 트레이너로 resume 하거나 \
                     저장 시 사용한 트레이너를 확인하세요.",
                    tag.as_str(),
                    expected.as_str()
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn save_model(
    directory: &str,
    paradigm: &str,
    completed: usize,
    schedule: EpochSchedule,
    loss: f32,
    lr: f32,
    seed: u64,
    save: impl FnOnce(&Path) -> MlResult<()>,
) -> MlResult<CheckpointPaths> {
    let directory = Path::new(directory);
    std::fs::create_dir_all(directory).map_err(|e| MlError::StringError(e.to_string()))?;
    let model = directory.join("model.tdw");
    let metadata = directory.join("training.json");
    save(&model)?;
    let tag = match paradigm {
        "supervised" => ParadigmTag::Supervised,
        "unsupervised" => ParadigmTag::Unsupervised,
        "semi_supervised" => ParadigmTag::SemiSupervised,
        "autoregressive" => ParadigmTag::Autoregressive,
        _ => ParadigmTag::Reinforcement,
    };
    TrainingCheckpoint {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        epochs_done: completed,
        total_epochs: schedule.epochs,
        last_loss: loss,
        tolerance: schedule.convergence.tolerance(),
        optimizer_lr: lr,
        model_path: model.to_string_lossy().into_owned(),
        timestamp: time::OffsetDateTime::now_utc().to_string(),
        paradigm: Some(tag),
        rng_seed: seed,
        optimizer_snapshot: None,
    }
    .save(&metadata)?;
    Ok(CheckpointPaths { model, metadata })
}
