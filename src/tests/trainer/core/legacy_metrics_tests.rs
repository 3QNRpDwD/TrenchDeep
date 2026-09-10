use super::*;
use crate::legacy::tensor::Tensor;

#[test]
fn classification_accuracy_hook_matches_direct() {
    // 3 개 배치. 정답 예측 2 개, 오답 예측 1 개 → 66.67%.
    let preds  = [
        Tensor::from_vec(vec![0.1, 0.9, 0.0], &[1, 3]).unwrap(),  // 1
        Tensor::from_vec(vec![0.7, 0.2, 0.1], &[1, 3]).unwrap(),  // 0
        Tensor::from_vec(vec![0.2, 0.3, 0.5], &[1, 3]).unwrap(),  // 2
    ];
    let targets = [
        Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(),  // 1 ✓
        Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]).unwrap(),  // 0 ✓
        Tensor::from_vec(vec![0.0, 1.0, 0.0], &[1, 3]).unwrap(),  // 1 ✗
    ];

    // 직접 경로
    let mut direct = ClassificationAccuracy::new();
    for (p, t) in preds.iter().zip(targets.iter()) {
        direct.update(p, t);
    }

    // 훅 경로
    let mut hook = ClassificationAccuracy::new();
    for (i, (p, t)) in preds.iter().zip(targets.iter()).enumerate() {
        let ctx = BatchContext {
            batch_idx: i,
            pred:      Some(p as &dyn TensorBase),
            target:    Some(t as &dyn TensorBase),
            loss:      0.0,
            n_tokens:  None,
            lambda:    None,
            lr:        1e-3,
        };
        <ClassificationAccuracy as MetricHook>::update(&mut hook, &ctx);
    }

    assert!((direct.compute() - MetricHook::compute(&hook)).abs() < 1e-6);
    assert!((MetricHook::compute(&hook) - 200.0 / 3.0).abs() < 1e-4);
}

#[test]
fn perplexity_hook_weighted_by_tokens() {
    // 두 배치: loss 1.0 @ 10 tokens, loss 2.0 @ 20 tokens
    // mean_nll = (1*10 + 2*20) / 30 = 50/30
    let mut hook = Perplexity::new();
    for (loss, n) in [(1.0_f32, 10usize), (2.0_f32, 20usize)] {
        let ctx = BatchContext {
            batch_idx: 0,
            pred:      None,
            target:    None,
            loss,
            n_tokens:  Some(n),
            lambda:    None,
            lr:        1e-3,
        };
        <Perplexity as MetricHook>::update(&mut hook, &ctx);
    }
    let expected = (50.0_f32 / 30.0).exp();
    let actual   = MetricHook::compute(&hook);
    assert!((actual - expected).abs() < 1e-4, "expected {}, got {}", expected, actual);
}

#[test]
fn classification_accuracy_hook_skips_when_target_missing() {
    // target 이 None 이면 호출이 no-op 이어야 함 (비지도/AR 에 훅이 붙어도 안전).
    let mut hook = ClassificationAccuracy::new();
    let pred = Tensor::from_vec(vec![0.1, 0.9, 0.0], &[1, 3]).unwrap();
    let ctx = BatchContext {
        batch_idx: 0,
        pred:      Some(&pred as &dyn TensorBase),
        target:    None,
        loss:      0.0,
        n_tokens:  None,
        lambda:    None,
        lr:        1e-3,
    };
    <ClassificationAccuracy as MetricHook>::update(&mut hook, &ctx);
    assert_eq!(MetricHook::compute(&hook), 0.0);
}
