use super::*;

// checkpoint 전용 import
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::path::Path;

pub const CHECKPOINT_SCHEMA_VERSION: u32 = 2;
fn default_schema_version() -> u32 { 1 }

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
// ParadigmTag — 체크포인트가 어느 패러다임에서 저장되었는지 식별
// ────────────────────────────────────────────────────────────────────────────

/// 체크포인트가 생성된 트레이너 패러다임. `resume()` 시 교차 로드를 방지한다.
///
/// 예를 들어 `SupervisedTrainer::resume()` 에 AR 체크포인트를 넘기면 모델
/// 인터페이스와 데이터 시맨틱이 맞지 않아 실행 중 이상 동작으로 이어진다.
/// 여기서 미리 태그 불일치를 탐지해 명확한 에러로 실패시킨다.
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
            ParadigmTag::Supervised      => "supervised",
            ParadigmTag::Unsupervised    => "unsupervised",
            ParadigmTag::SemiSupervised  => "semi-supervised",
            ParadigmTag::Autoregressive  => "autoregressive",
            ParadigmTag::Reinforcement   => "reinforcement",
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
    pub optimizer_snapshot: Option<crate::optimizer::OptimizerSnapshot>,
}

impl TrainingCheckpoint {
    /// 체크포인트를 JSON 파일로 저장한다.
    pub fn save(&self, path: impl AsRef<Path>) -> MlResult<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
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
                format!("체크포인트 저장 실패 '{}': {}", path.display(), e)
            ))
    }

    /// JSON 파일에서 체크포인트를 로드.
    pub fn load(path: impl AsRef<Path>) -> MlResult<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)
            .map_err(|e| MlError::StringError(
                format!("체크포인트 로드 실패 '{}': {}", path.display(), e)
            ))?;
        serde_json::from_str(&json)
            .map_err(|e| MlError::StringError(
                format!("체크포인트 파싱 실패 '{}': {}", path.display(), e)
            ))
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
                    tag.as_str(), expected.as_str()
                )));
            }
        }
        Ok(())
    }
}

pub struct CheckpointManager;

impl CheckpointManager {
    pub fn load_into<M: CheckpointableModel>(
        metadata: impl AsRef<Path>, expected: ParadigmTag, model: &mut M,
        optimizer: &mut dyn crate::optimizer::Optimizer,
    ) -> MlResult<TrainingCheckpoint> {
        let ckpt = TrainingCheckpoint::load(metadata)?;
        ckpt.verify_paradigm(expected)?;
        model.load_checkpoint(Path::new(&ckpt.model_path))?;
        optimizer.set_lr(ckpt.optimizer_lr);
        // TODO(Phase-6): restore optional optimizer_snapshot after registration validation.
        Ok(ckpt)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 공용 저장 헬퍼 — 각 트레이너의 `fit_inner` 에서 중복 제거용
// ────────────────────────────────────────────────────────────────────────────

/// 인터럽트 시 모델+체크포인트를 저장하고 UX(progress bar / info 로그) 를
/// 마무리한다. 각 패러다임별 `fit_inner` 에 복사-붙여넣기 되어 있던 블록을
/// 하나의 엔트리포인트로 통합한다.
///
/// # 동작
/// 1. `ckpt_dir/model_weights.tdw` 에 `save_model` 호출로 가중치 저장.
/// 2. 위가 성공하면 `ckpt_dir/checkpoint.json` 에 `TrainingCheckpoint` 직렬화.
/// 3. 각 단계의 성공/실패를 progress bar 와 logging 에 반영.
///
/// `save_model` 은 `FnOnce(&str) -> MlResult<()>` 로 전달한다. 호출자가 보통
/// `|p| model.save_model(p)` 형태로 전달한다.
pub(crate) fn save_interrupt_checkpoint<F>(
    ckpt_dir:     &Path,
    epochs_done:  usize,
    total_epochs: usize,
    loss:         f32,
    tolerance:    f32,
    optimizer_lr: f32,
    paradigm:     ParadigmTag,
    rng_seed:     u64,
    save_model:   F,
    progress:     &super::progress::EpochProgress,
) -> MlResult<CheckpointPaths> where
    F: FnOnce(&Path) -> MlResult<()>,
{
    std::fs::create_dir_all(ckpt_dir)
        .map_err(|e| MlError::StringError(format!("checkpoint directory creation failed: {e}")))?;
    let model_path = ckpt_dir.join("model_weights.tdw");
    let ckpt_path = ckpt_dir.join("checkpoint.json");
    let model_tmp = ckpt_dir.join("model_weights.tdw.tmp");
    let ckpt_tmp = ckpt_dir.join("checkpoint.json.tmp");

    save_model(&model_tmp)?;
    let ckpt = TrainingCheckpoint {
                schema_version: CHECKPOINT_SCHEMA_VERSION,
                epochs_done,
                total_epochs,
                last_loss: loss,
                tolerance,
                optimizer_lr,
                model_path: model_path.to_string_lossy().into_owned(),
                timestamp: format!("{:?}", std::time::SystemTime::now()),
                paradigm: Some(paradigm),
                rng_seed,
                optimizer_snapshot: None,
            };
    if let Err(e) = ckpt.save(&ckpt_tmp) {
        let _ = std::fs::remove_file(&model_tmp);
        return Err(e);
    }
    replace_file(&model_tmp, &model_path)?;
    replace_file(&ckpt_tmp, &ckpt_path)?;
    progress.finish_interrupted();
    Ok(CheckpointPaths { model: model_path, metadata: ckpt_path })
}

fn replace_file(source: &Path, target: &Path) -> MlResult<()> {
    if target.exists() {
        let backup = target.with_extension("bak");
        if backup.exists() { std::fs::remove_file(&backup).ok(); }
        std::fs::rename(target, &backup)
            .map_err(|e| MlError::StringError(format!("checkpoint backup failed: {e}")))?;
        if let Err(e) = std::fs::rename(source, target) {
            let _ = std::fs::rename(&backup, target);
            return Err(MlError::StringError(format!("checkpoint replace failed: {e}")));
        }
        let _ = std::fs::remove_file(backup);
    } else {
        std::fs::rename(source, target)
            .map_err(|e| MlError::StringError(format!("checkpoint commit failed: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> String {
        let dir = std::env::temp_dir();
        let unique = format!("{:?}", std::time::SystemTime::now())
            .chars().filter(|c| c.is_alphanumeric()).collect::<String>();
        format!("{}/trench_deep_ckpt_{}_{}.json", dir.display(), unique, name)
    }

    fn sample(paradigm: Option<ParadigmTag>) -> TrainingCheckpoint {
        TrainingCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            epochs_done: 3,
            total_epochs: 10,
            last_loss: 0.1234,
            tolerance: 1e-6,
            optimizer_lr: 1e-3,
            model_path: "ignore".into(),
            timestamp: "t0".into(),
            paradigm,
            rng_seed: 42,
            optimizer_snapshot: None,
        }
    }

    #[test]
    fn roundtrip_preserves_all_fields() {
        let path = tmp_path("rt");
        let ckpt = sample(Some(ParadigmTag::Supervised));
        ckpt.save(&path).expect("save");
        let loaded = TrainingCheckpoint::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.epochs_done,  ckpt.epochs_done);
        assert_eq!(loaded.total_epochs, ckpt.total_epochs);
        assert!((loaded.last_loss - ckpt.last_loss).abs() < 1e-6);
        assert_eq!(loaded.paradigm,     Some(ParadigmTag::Supervised));
        assert_eq!(loaded.rng_seed,     42);
    }

    #[test]
    fn verify_paradigm_catches_mismatch() {
        let ckpt = sample(Some(ParadigmTag::Supervised));
        assert!(ckpt.verify_paradigm(ParadigmTag::Supervised).is_ok());
        let err = ckpt.verify_paradigm(ParadigmTag::Autoregressive)
            .expect_err("should refuse paradigm mismatch");
        let msg = format!("{:?}", err);
        assert!(msg.contains("supervised") && msg.contains("autoregressive"),
                "에러 메시지에 두 패러다임 이름이 포함되어야 함: {}", msg);
    }

    #[test]
    fn verify_paradigm_allows_legacy_untagged() {
        // Phase 3 이전 체크포인트는 paradigm=None 으로 로드됨 → 통과해야 함.
        let ckpt = sample(None);
        assert!(ckpt.verify_paradigm(ParadigmTag::Supervised).is_ok());
        assert!(ckpt.verify_paradigm(ParadigmTag::Reinforcement).is_ok());
    }

    #[test]
    fn legacy_checkpoint_loads_without_paradigm_field() {
        // 구버전 포맷: paradigm / rng_seed 필드가 JSON 에 아예 없어야 로드 가능.
        let path = tmp_path("legacy");
        let legacy_json = r#"{
            "epochs_done": 5,
            "total_epochs": 20,
            "last_loss": 0.5,
            "tolerance": 1e-6,
            "optimizer_lr": 0.001,
            "model_path": "foo",
            "timestamp": "t"
        }"#;
        std::fs::write(&path, legacy_json).unwrap();
        let loaded = TrainingCheckpoint::load(&path).expect("legacy 로드 실패");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.epochs_done, 5);
        assert_eq!(loaded.paradigm,    None);
        assert_eq!(loaded.rng_seed,    0);
    }

    #[test]
    fn failed_model_save_preserves_existing_checkpoint() {
        let dir = std::env::temp_dir().join(format!("trench_atomic_{:?}", std::time::SystemTime::now())
            .replace(':', "_").replace(' ', "_"));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model_weights.tdw");
        let meta = dir.join("checkpoint.json");
        std::fs::write(&model, b"old-model").unwrap();
        std::fs::write(&meta, b"old-meta").unwrap();
        let progress = crate::trainer::progress::EpochProgress::new(1, false);
        let result = save_interrupt_checkpoint(&dir, 1, 2, 0.5, 0.0, 0.1,
            ParadigmTag::Supervised, 7,
            |_path| Err(MlError::StringError("injected failure".into())), &progress);
        assert!(result.is_err());
        assert_eq!(std::fs::read(&model).unwrap(), b"old-model");
        assert_eq!(std::fs::read(&meta).unwrap(), b"old-meta");
        std::fs::remove_dir_all(dir).ok();
    }
}
