use super::*;

#[test]
fn sample_categorical_respects_distribution() {
    // logits = [-10, 10] → 거의 확실하게 action=1
    let mut count_one = 0;
    for i in 0..200 {
        if sample_categorical(&[-10.0, 10.0], (i as f32 + 0.5) / 200.0) == 1 {
            count_one += 1;
        }
    }
    assert!(count_one > 190, "action 1 이 거의 항상 선택되어야 함: {}/200", count_one);
}

#[test]
fn sample_categorical_handles_edge_cases() {
    // 동일 로짓 → 어떤 값도 반환 가능하지만 panic 하면 안 됨
    for i in 0..50 {
        let a = sample_categorical(&[0.0, 0.0, 0.0], i as f32 / 50.0);
        assert!(a < 3);
    }
    // 단일 원소
    assert_eq!(sample_categorical(&[5.0], 0.5), 0);
    // 비정상 로짓
    assert!(sample_categorical(&[f32::NAN, 1.0], 0.5) < 2);
}
