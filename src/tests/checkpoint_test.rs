use crate::{
    MlResult,
    nn::{Linear, Sequential, GroupNormLayer, Parameter, Layer},
    tensor::{Tensor, TensorBase},
};

// ── 헬퍼 ────────────────────────────────────────────────────────────────────

fn build_net() -> MlResult<Sequential> {
    let mut net = Sequential::new("test_net");
    net.push(Box::new(Linear::new(4, 8, "fc1")?));
    net.push(Box::new(Linear::new(8, 2, "fc2")?));
    Ok(net)
}

fn weights_of(net: &Sequential) -> Vec<Vec<f32>> {
    net.params().iter().map(|p| p.tensor().data().to_vec()).collect()
}

fn zero_all_weights(net: &mut Sequential) {
    for p in net.params() {
        let shape = p.tensor().shape().to_vec();
        let zeros = crate::tensor::GlobalTensor::zeros(&shape);
        p.tensor().replace(zeros);
    }
}

// ── JSON 저장/로드 테스트 ─────────────────────────────────────────────────────

#[test]
fn test_sequential_save_load_json() -> MlResult<()> {
    let tmp = "target/test_checkpoints/sequential_test.json";

    let net = build_net()?;
    let original_weights = weights_of(&net);

    // 저장
    net.save(tmp)?;
    assert!(std::path::Path::new(tmp).exists(), "JSON 파일이 생성되어야 함");

    // 로드: 가중치를 0으로 초기화한 뒤 복원
    let mut net2 = build_net()?;
    zero_all_weights(&mut net2);
    net2.load(tmp)?;

    let loaded_weights = weights_of(&net2);
    assert_eq!(original_weights.len(), loaded_weights.len(), "파라미터 개수가 일치해야 함");
    for (orig, loaded) in original_weights.iter().zip(loaded_weights.iter()) {
        assert_eq!(orig, loaded, "가중치 값이 일치해야 함");
    }

    std::fs::remove_file(tmp).ok();
    Ok(())
}

// ── binary (.tdw) 저장/로드 테스트 ──────────────────────────────────────────

#[test]
fn test_sequential_save_load_binary() -> MlResult<()> {
    let tmp = "target/test_checkpoints/sequential_test.tdw";

    let net = build_net()?;
    let original_weights = weights_of(&net);

    net.save(tmp)?;
    assert!(std::path::Path::new(tmp).exists(), ".tdw 파일이 생성되어야 함");

    let mut net2 = build_net()?;
    zero_all_weights(&mut net2);
    net2.load(tmp)?;

    let loaded_weights = weights_of(&net2);
    for (orig, loaded) in original_weights.iter().zip(loaded_weights.iter()) {
        assert_eq!(orig, loaded, "가중치 값이 일치해야 함 (binary 포맷)");
    }

    std::fs::remove_file(tmp).ok();
    Ok(())
}

// ── shape 불일치 에러 테스트 ────────────────────────────────────────────────

#[test]
fn test_load_shape_mismatch_returns_error() -> MlResult<()> {
    let tmp = "target/test_checkpoints/shape_mismatch.json";

    // 4→8→2 구조로 저장
    let net = build_net()?;
    net.save(tmp)?;

    // 4→16→2 구조에 로드 시도 → shape 불일치 에러
    let mut net_wrong = Sequential::new("test_net");
    net_wrong.push(Box::new(Linear::new(4, 16, "fc1")?)); // fc1 shape 다름
    net_wrong.push(Box::new(Linear::new(16, 2, "fc2")?));

    let result = net_wrong.load(tmp);
    assert!(result.is_err(), "shape 불일치 시 에러가 반환되어야 함");

    std::fs::remove_file(tmp).ok();
    Ok(())
}

// ── 레이블 불일치 시 경고 후 건너뜀 ─────────────────────────────────────────

#[test]
fn test_load_missing_label_skips_gracefully() -> MlResult<()> {
    let tmp = "target/test_checkpoints/label_mismatch.json";

    let net = build_net()?;
    let original_fc2 = net.params()[2].tensor().data().to_vec(); // fc2 weight
    net.save(tmp)?;

    // fc1 레이블을 fc1_renamed으로 바꾼 구조로 로드
    // → fc1_renamed은 파일에 없으므로 건너뜀, fc2는 정상 복원
    let mut net2 = Sequential::new("test_net");
    net2.push(Box::new(Linear::new(4, 8, "fc1_renamed")?));
    net2.push(Box::new(Linear::new(8, 2, "fc2")?));

    // fc2_weight를 0으로 초기화 후 복원 확인
    let fc2_weight = net2.params()[2].tensor();
    fc2_weight.replace(crate::tensor::GlobalTensor::zeros(fc2_weight.shape()));

    let result = net2.load(tmp); // 에러 없이 성공해야 함
    assert!(result.is_ok(), "레이블 불일치 시에도 에러 없이 진행해야 함");

    let loaded_fc2 = net2.params()[2].tensor().data().to_vec();
    assert_eq!(original_fc2, loaded_fc2, "fc2는 정상 복원되어야 함");

    std::fs::remove_file(tmp).ok();
    Ok(())
}

// ── GroupNormLayer 저장/로드 테스트 ─────────────────────────────────────────

#[test]
fn test_group_norm_save_load() -> MlResult<()> {
    let tmp = "target/test_checkpoints/group_norm.json";

    let mut net = Sequential::new("gn_net");
    net.push(Box::new(GroupNormLayer::new(4, 8, 1e-5, "gn1")?));

    // gamma를 2.0, beta를 0.5로 설정
    let params = net.params();
    let custom_gamma = crate::tensor::GlobalTensor::from_vec(vec![2.0f32; 8], &[8])?;
    let custom_beta  = crate::tensor::GlobalTensor::from_vec(vec![0.5f32; 8], &[8])?;
    params[0].tensor().replace(custom_gamma);
    params[1].tensor().replace(custom_beta);
    drop(params);

    net.save(tmp)?;

    let mut net2 = Sequential::new("gn_net");
    net2.push(Box::new(GroupNormLayer::new(4, 8, 1e-5, "gn1")?));
    net2.load(tmp)?;

    let p = net2.params();
    assert!(p[0].tensor().data().iter().all(|&v| (v - 2.0).abs() < 1e-6), "gamma 복원 실패");
    assert!(p[1].tensor().data().iter().all(|&v| (v - 0.5).abs() < 1e-6), "beta 복원 실패");

    std::fs::remove_file(tmp).ok();
    Ok(())
}

// ── JSON 내용 구조 검증 ────────────────────────────────────────────────────

#[test]
fn test_json_structure() -> MlResult<()> {
    let tmp = "target/test_checkpoints/structure.json";

    let net = build_net()?;
    net.save(tmp)?;

    let json = std::fs::read_to_string(tmp).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["version"], "1", "version 필드가 있어야 함");
    assert!(v["layers"].is_array(), "layers 배열이 있어야 함");
    assert_eq!(v["layers"][0]["layer_type"], "Linear");
    assert_eq!(v["layers"][0]["label"], "fc1");
    assert!(v["layers"][0]["config"]["in_features"].is_number());
    assert!(v["layers"][0]["params"][0]["data"].is_array(), "weight data가 있어야 함");

    std::fs::remove_file(tmp).ok();
    Ok(())
}
