use crate::tensor::TensorBase;

/// shape + 앞 최대 4개 값 + NaN/Inf 여부를 포함한 요약 문자열 반환
pub fn summary(label: &str, t: &dyn TensorBase) -> String {
    let data = t.data();
    let preview: Vec<f32> = data.iter().take(4).copied().collect();
    let has_nan = data.iter().any(|x| x.is_nan());
    let has_inf = data.iter().any(|x| x.is_infinite());
    format!(
        "[{}] shape={:?} preview={:?}{}{}",
        label,
        t.shape(),
        preview,
        if has_nan { " ⚠NaN" } else { "" },
        if has_inf { " ⚠Inf" } else { "" },
    )
}

/// shape + 앞 최대 4개 값 + NaN/Inf 여부 (GlobalTensor 전용 - data/shape 직접 접근)
pub fn summary_raw(label: &str, data: &[f32], shape: &[usize]) -> String {
    let preview: Vec<f32> = data.iter().take(4).copied().collect();
    let has_nan = data.iter().any(|x| x.is_nan());
    let has_inf = data.iter().any(|x| x.is_infinite());
    format!(
        "[{}] shape={:?} preview={:?}{}{}",
        label,
        shape,
        preview,
        if has_nan { " ⚠NaN" } else { "" },
        if has_inf { " ⚠Inf" } else { "" },
    )
}

/// min / max / mean / L2 norm 통계를 trace 레벨로 출력
pub fn stats(label: &str, t: &dyn TensorBase) {
    let data = t.data();
    if data.is_empty() {
        tracing::trace!("[{}] (empty)", label);
        return;
    }
    let min  = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max  = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    tracing::trace!(
        "[{}] shape={:?} min={:.4} max={:.4} mean={:.4} norm={:.4}",
        label, t.shape(), min, max, mean, norm
    );
}

/// min / max / mean / L2 norm 통계 (GlobalTensor 전용 - data/shape 직접 접근)
pub fn stats_raw(label: &str, data: &[f32], shape: &[usize]) {
    if data.is_empty() {
        tracing::trace!("[{}] (empty)", label);
        return;
    }
    let min  = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max  = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    tracing::trace!(
        "[{}] shape={:?} min={:.4} max={:.4} mean={:.4} norm={:.4}",
        label, shape, min, max, mean, norm
    );
}
