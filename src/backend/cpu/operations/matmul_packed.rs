//! Allocation-free GEMM for prepared execution. Scratch belongs to the executor.
const MC: usize = 32;
const KC: usize = 128;
const NC: usize = 128;
pub(super) const WORK: usize = KC * NC + MC * KC;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_special_values_preserve_scalar_classification() {
        let (m, k, n) = (17, 131, 33);
        let mut a = vec![0.5; m * k];
        let mut b = vec![-0.0; k * n];
        a[0] = f32::INFINITY;
        a[k + 1] = f32::NEG_INFINITY;
        a[2 * k + 2] = f32::NAN;
        b[n + 1] = 1.0;
        b[2 * n + 2] = -1.0;
        let mut c = vec![0.0; m * n];
        let mut reference = c.clone();
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    reference[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        gemm(
            &a,
            &b,
            &mut c,
            m,
            k,
            n,
            k,
            1,
            n,
            1,
            &mut vec![f32::NAN; WORK],
        );
        for (actual, expected) in c.into_iter().zip(reference) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
    #[test]
    fn strided_panels_match_f64_oracle() {
        for (m, k, n) in [(3, 7, 5), (17, 131, 137), (65, 129, 33)] {
            for trans_a in [false, true] {
                for trans_b in [false, true] {
                    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.17).sin()).collect();
                    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.13).cos()).collect();
                    let (ar, ac) = if trans_a { (1, m) } else { (k, 1) };
                    let (br, bc) = if trans_b { (1, k) } else { (n, 1) };
                    let mut c = vec![0.25; m * n];
                    let mut work = vec![f32::NAN; WORK];
                    gemm(&a, &b, &mut c, m, k, n, ar, ac, br, bc, &mut work);
                    for i in 0..m {
                        for j in 0..n {
                            let expected = 0.25
                                + (0..k)
                                    .map(|p| a[i * ar + p * ac] as f64 * b[p * br + j * bc] as f64)
                                    .sum::<f64>();
                            assert!(
                                (c[i * n + j] as f64 - expected).abs()
                                    <= 1e-5 + 1e-4 * expected.abs()
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Accumulate A*B into row-major C; strides describe logical transposes.
pub(super) fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    ar: usize,
    ac: usize,
    br: usize,
    bc: usize,
    scratch: &mut [f32],
) {
    if m.saturating_mul(k).saturating_mul(n) < 32768 {
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * ar + p * ac] * b[p * br + j * bc];
                }
            }
        }
        return;
    }
    let (scratch, packed_a) = scratch.split_at_mut(KC * NC);
    for j0 in (0..n).step_by(NC) {
        let nn = (n - j0).min(NC);
        for p0 in (0..k).step_by(KC) {
            let kk = (k - p0).min(KC);
            for p in 0..kk {
                for j in 0..nn {
                    scratch[p * nn + j] = b[(p0 + p) * br + (j0 + j) * bc];
                }
            }
            for i0 in (0..m).step_by(MC) {
                let mm = (m - i0).min(MC);
                for i in 0..mm {
                    for p in 0..kk {
                        packed_a[i * kk + p] = a[(i0 + i) * ar + (p0 + p) * ac];
                    }
                }
                let a = &packed_a[..mm * kk];
                for i in 0..mm {
                    let row = &mut c[(i0 + i) * n + j0..(i0 + i) * n + j0 + nn];
                    let mut j = 0;
                    #[cfg(target_arch = "x86_64")]
                    if std::arch::is_x86_feature_detected!("avx2") {
                        while j + 32 <= nn {
                            // Four independent vector accumulators hide add latency.
                            unsafe {
                                thirty_two(a, scratch, row, i * kk, nn, kk, j);
                            }
                            j += 32;
                        }
                        while j + 8 <= nn {
                            // All eight C/B lanes and all A elements lie in validated slices.
                            unsafe {
                                eight(a, scratch, row, i * kk, 1, nn, kk, j);
                            }
                            j += 8;
                        }
                    }
                    for col in j..nn {
                        let mut sum = row[col];
                        for p in 0..kk {
                            sum += a[i * kk + p] * scratch[p * nn + col];
                        }
                        row[col] = sum;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn thirty_two(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    base: usize,
    n: usize,
    k: usize,
    j: usize,
) {
    use std::arch::x86_64::*;
    // Caller guarantees 32 destination lanes, k A values and k packed B rows.
    unsafe {
        let mut x0 = _mm256_loadu_ps(c.as_ptr().add(j));
        let mut x1 = _mm256_loadu_ps(c.as_ptr().add(j + 8));
        let mut x2 = _mm256_loadu_ps(c.as_ptr().add(j + 16));
        let mut x3 = _mm256_loadu_ps(c.as_ptr().add(j + 24));
        for p in 0..k {
            let av = _mm256_set1_ps(a[base + p]);
            let bp = b.as_ptr().add(p * n + j);
            x0 = _mm256_add_ps(x0, _mm256_mul_ps(av, _mm256_loadu_ps(bp)));
            x1 = _mm256_add_ps(x1, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(8))));
            x2 = _mm256_add_ps(x2, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(16))));
            x3 = _mm256_add_ps(x3, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(24))));
        }
        _mm256_storeu_ps(c.as_mut_ptr().add(j), x0);
        _mm256_storeu_ps(c.as_mut_ptr().add(j + 8), x1);
        _mm256_storeu_ps(c.as_mut_ptr().add(j + 16), x2);
        _mm256_storeu_ps(c.as_mut_ptr().add(j + 24), x3);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eight(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    base: usize,
    stride: usize,
    n: usize,
    k: usize,
    j: usize,
) {
    use std::arch::x86_64::*;
    // Separate multiply/add preserves the existing reduction order, including across tiles.
    unsafe {
        let mut value = _mm256_loadu_ps(c.as_ptr().add(j));
        for p in 0..k {
            let av = _mm256_set1_ps(a[base + p * stride]);
            let bv = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
            value = _mm256_add_ps(value, _mm256_mul_ps(av, bv));
        }
        _mm256_storeu_ps(c.as_mut_ptr().add(j), value);
    }
}
