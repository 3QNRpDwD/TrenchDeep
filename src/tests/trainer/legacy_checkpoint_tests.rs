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
    let progress = crate::legacy::trainer::progress::EpochProgress::new(1, false);
    let result = save_interrupt_checkpoint(&dir, 1, 2, 0.5, 0.0, 0.1,
        ParadigmTag::Supervised, 7,
        |_path| Err(MlError::StringError("injected failure".into())), &progress);
    assert!(result.is_err());
    assert_eq!(std::fs::read(&model).unwrap(), b"old-model");
    assert_eq!(std::fs::read(&meta).unwrap(), b"old-meta");
    std::fs::remove_dir_all(dir).ok();
}
