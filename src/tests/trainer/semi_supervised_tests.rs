use super::*;

#[test]
fn consistency_ramp_constant() {
    let r = ConsistencyRamp::Constant(0.5);
    assert_eq!(r.value(0), 0.5);
    assert_eq!(r.value(100), 0.5);
}

#[test]
fn consistency_ramp_sigmoid_monotonic() {
    let r = ConsistencyRamp::Sigmoid {
        max_weight: 1.0,
        ramp_epochs: 30,
    };
    let v0 = r.value(0);
    let v15 = r.value(15);
    let v30 = r.value(30);
    let v40 = r.value(40);
    assert!(v0 < v15, "0 < 15 실패: {} < {}", v0, v15);
    assert!(v15 < v30, "15 < 30 실패: {} < {}", v15, v30);
    assert!(
        (v30 - 1.0).abs() < 1e-5,
        "램프 끝에서 max_weight 수렴 실패: {}",
        v30
    );
    assert!(
        (v40 - 1.0).abs() < 1e-5,
        "램프 초과 영역에서 max_weight 유지 실패: {}",
        v40
    );
}

#[test]
fn consistency_ramp_sigmoid_zero_length() {
    let r = ConsistencyRamp::Sigmoid {
        max_weight: 0.7,
        ramp_epochs: 0,
    };
    assert_eq!(r.value(0), 0.7);
}
