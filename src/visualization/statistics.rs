use super::TensorStatistics;

#[cfg(test)]
thread_local! {
    static COLLECTION_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn tensor_statistics(data: &[f32]) -> Option<TensorStatistics> {
    #[cfg(test)]
    COLLECTION_COUNT.with(|count| count.set(count.get() + 1));
    if data.is_empty() {
        return None;
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut squared = 0.0f64;
    let mut l1 = 0.0f64;
    let mut finite = 0usize;
    let mut zeros = 0usize;
    let mut nan = 0usize;
    let mut positive_infinity = 0usize;
    let mut negative_infinity = 0usize;

    for &value in data {
        if value.is_nan() {
            nan += 1;
            continue;
        }
        if value == f32::INFINITY {
            positive_infinity += 1;
            continue;
        }
        if value == f32::NEG_INFINITY {
            negative_infinity += 1;
            continue;
        }
        min = min.min(value);
        max = max.max(value);
        let value = value as f64;
        sum += value;
        squared += value * value;
        l1 += value.abs();
        finite += 1;
        if value == 0.0 {
            zeros += 1;
        }
    }

    let mean = (finite > 0).then(|| (sum / finite as f64) as f32);
    let variance =
        mean.map(|mean| (squared / finite as f64 - (mean as f64).powi(2)).max(0.0) as f32);
    let finite_value = |value: f64| value.is_finite().then_some(value as f32);
    Some(TensorStatistics {
        min: (finite > 0).then_some(min),
        max: (finite > 0).then_some(max),
        mean,
        std_dev: variance.map(f32::sqrt),
        l1_norm: finite_value(l1),
        l2_norm: finite_value(squared.sqrt()),
        zeros,
        nan,
        positive_infinity,
        negative_infinity,
    })
}

#[cfg(test)]
pub(crate) fn reset_collection_count() {
    COLLECTION_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn collection_count() -> usize {
    COLLECTION_COUNT.with(|count| count.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_classify_non_finite_and_zero_values() {
        let stats =
            tensor_statistics(&[-2.0, 0.0, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
                .unwrap();
        assert_eq!(stats.min, Some(-2.0));
        assert_eq!(stats.max, Some(2.0));
        assert_eq!(stats.mean, Some(0.0));
        assert_eq!(stats.zeros, 1);
        assert_eq!(stats.nan, 1);
        assert_eq!(stats.positive_infinity, 1);
        assert_eq!(stats.negative_infinity, 1);
    }
}
