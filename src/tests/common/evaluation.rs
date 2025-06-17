use super::*;

/// 테스트 데이터셋으로 모델의 정확도를 평가하고 결과를 반환합니다.
pub fn evaluate_model(mlp: &dyn Model, x_test: &[Arc<Variable>], t_test: &[Arc<Variable>]) -> MlResult<f32> {
    let n_val = x_test.len();
    info!("Starting model evaluation on {} test samples...", n_val);

    let mut correct_predictions = 0;
    for i in 0..n_val {
        let test_input = &x_test[i];
        let true_label_tensor = &t_test[i];

        let y = mlp.predict(test_input.tensor())?;

        // 예측 결과에서 가장 확률이 높은 클래스의 인덱스를 찾습니다.
        let predicted_class = y.data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);

        // 실제 레이블(one-hot)에서 정답 클래스의 인덱스를 찾습니다.
        let true_class = true_label_tensor.tensor().data()
            .iter()
            .position(|&r| r == 1.0)
            .unwrap_or(0);

        if predicted_class == true_class {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / n_val as f32 * 100.0;
    info!("✅ Evaluation complete: Accuracy = {:.2}%", accuracy);

    Ok(accuracy)
}