//! NumPy 스타일 브로드캐스트 유틸리티.
//!
//! 모든 공개 함수는 **flat offset** 기반으로 동작합니다.
//! 텐서 데이터는 row-major (C-order) 로 저장된다고 가정합니다.

use crate::{MlError, MlResult, TensorError};

/// 두 shape 를 NumPy 규약으로 브로드캐스트한 결과 shape 를 반환합니다.
///
/// 규약:
/// 1. 더 짧은 shape 의 앞쪽에 1을 채워 rank 를 맞춘다.
/// 2. 각 축에서 `a == b || a == 1 || b == 1` 이어야 호환.
/// 3. 출력 축 크기는 `max(a, b)`.
///
/// 호환 불가능한 shape 쌍은 `TensorError::InvalidShape` 로 반환.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> MlResult<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let ai = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let bi = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        if ai == bi || ai == 1 || bi == 1 {
            out.push(ai.max(bi));
        } else {
            return Err(MlError::TensorError(TensorError::InvalidShape {
                expected: a.to_vec(),
                got: b.to_vec(),
            }));
        }
    }
    Ok(out)
}

/// 출력 좌표 (row-major flat index) 에 대응하는 a / b 의 flat offset 을 계산합니다.
///
/// 결과 벡터 길이는 `out_shape.iter().product()`.
/// 각 원소는 `(a_offset, b_offset)`.
pub fn broadcast_offsets(
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<(usize, usize)> {
    let rank = out_shape.len();
    let total: usize = out_shape.iter().product();

    // leading 1 으로 패딩한 shape
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1usize; rank - s.len()];
        v.extend_from_slice(s);
        v
    };
    let a_pad = pad(a_shape);
    let b_pad = pad(b_shape);

    // stride (row-major) — broadcast 축(크기 1)은 stride 0 으로.
    let compute_stride = |shape_padded: &[usize], orig: &[usize]| -> Vec<usize> {
        let mut stride = vec![0usize; rank];
        if !orig.is_empty() {
            let mut s = 1usize;
            for i in (0..rank).rev() {
                if shape_padded[i] == 1 {
                    stride[i] = 0; // broadcast 축
                } else {
                    stride[i] = s;
                    s *= shape_padded[i];
                }
            }
        }
        stride
    };
    let a_stride = compute_stride(&a_pad, a_shape);
    let b_stride = compute_stride(&b_pad, b_shape);

    // 출력 stride
    let mut out_stride = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }

    let mut result = Vec::with_capacity(total);
    let mut coords = vec![0usize; rank];
    for flat in 0..total {
        // flat → coords
        let mut rem = flat;
        for i in 0..rank {
            coords[i] = rem / out_stride[i];
            rem %= out_stride[i];
        }
        // coords → a/b offset
        let mut ao = 0usize;
        let mut bo = 0usize;
        for i in 0..rank {
            ao += coords[i] * a_stride[i];
            bo += coords[i] * b_stride[i];
        }
        result.push((ao, bo));
    }
    result
}

/// `from` shape 로 broadcast 된 grad 를 `to` shape 로 sum-reduce.
///
/// `to` 는 `from` 의 broadcast 소스 중 하나여야 합니다
/// (broadcast_shape(to, ?) == from 이 성립).
pub fn reduce_to_shape(data: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return data.to_vec();
    }
    let rank = from.len();
    // to 를 leading 1 으로 패딩
    let mut to_pad = vec![1usize; rank - to.len()];
    to_pad.extend_from_slice(to);

    // 결과 크기
    let out_size: usize = to.iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; out_size];

    // stride 계산: 출력은 원본 to (패딩 제거) 기준. to_pad 에서 축이 1인 경우 stride=0.
    let mut to_stride_padded = vec![0usize; rank];
    let mut s = 1usize;
    for i in (0..rank).rev() {
        if to_pad[i] == 1 {
            to_stride_padded[i] = 0;
        } else {
            to_stride_padded[i] = s;
            s *= to_pad[i];
        }
    }

    // from stride
    let mut from_stride = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        from_stride[i] = from_stride[i + 1] * from[i + 1];
    }

    let total: usize = from.iter().product();
    let mut coords = vec![0usize; rank];
    for flat in 0..total {
        let mut rem = flat;
        for i in 0..rank {
            coords[i] = rem / from_stride[i];
            rem %= from_stride[i];
        }
        let mut off = 0usize;
        for i in 0..rank {
            off += coords[i] * to_stride_padded[i];
        }
        out[off] += data[flat];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_identity() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn shape_bias_last_dim() {
        // NumPy 규약: [N,H,W,C] + [C]  (C 가 마지막 축)
        assert_eq!(
            broadcast_shape(&[2, 8, 8, 4], &[4]).unwrap(),
            vec![2, 8, 8, 4]
        );
    }

    #[test]
    fn shape_channel_bias_needs_reshape() {
        // [N,C,H,W] + [C] 는 NumPy 규약상 호환 불가 (마지막 축 C vs W).
        // channel-first bias 는 [1,C,1,1] 로 reshape 후 더해야 함.
        assert!(broadcast_shape(&[2, 4, 8, 8], &[4]).is_err());
    }

    #[test]
    fn shape_time_emb() {
        assert_eq!(
            broadcast_shape(&[2, 4, 8, 8], &[2, 4, 1, 1]).unwrap(),
            vec![2, 4, 8, 8]
        );
    }

    #[test]
    fn shape_per_channel() {
        assert_eq!(
            broadcast_shape(&[2, 4, 8, 8], &[1, 4, 1, 1]).unwrap(),
            vec![2, 4, 8, 8]
        );
    }

    #[test]
    fn shape_attention_mask() {
        assert_eq!(
            broadcast_shape(&[2, 9, 9], &[1, 9, 9]).unwrap(),
            vec![2, 9, 9]
        );
    }

    #[test]
    fn shape_incompatible() {
        assert!(broadcast_shape(&[3], &[4]).is_err());
    }

    #[test]
    fn shape_incompatible_middle() {
        assert!(broadcast_shape(&[2, 3, 4], &[2, 5, 4]).is_err());
    }

    #[test]
    fn offsets_scalar_plus_vec() {
        // [1] + [3] → [3]. a_offset 모두 0, b_offset = [0,1,2]
        let offsets = broadcast_offsets(&[1], &[3], &[3]);
        assert_eq!(offsets, vec![(0, 0), (0, 1), (0, 2)]);
    }

    #[test]
    fn offsets_vec_plus_row() {
        // [2,3] + [3] → [2,3].
        // output flat: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
        // a_offset: 0 1 2 3 4 5
        // b_offset: 0 1 2 0 1 2   (last dim broadcast)
        let offsets = broadcast_offsets(&[2, 3], &[3], &[2, 3]);
        assert_eq!(
            offsets,
            vec![(0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2)]
        );
    }

    #[test]
    fn reduce_roundtrip_bias() {
        // from=[2,3], to=[3]: 각 열 합
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let reduced = reduce_to_shape(&data, &[2, 3], &[3]);
        assert_eq!(reduced, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_time_emb_grad() {
        // from=[2,2,2,2], to=[2,2,1,1]: 마지막 두 축 합
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let reduced = reduce_to_shape(&data, &[2, 2, 2, 2], &[2, 2, 1, 1]);
        // (n=0,c=0): 1+2+3+4=10
        // (n=0,c=1): 5+6+7+8=26
        // (n=1,c=0): 9+10+11+12=42
        // (n=1,c=1): 13+14+15+16=58
        assert_eq!(reduced, vec![10.0, 26.0, 42.0, 58.0]);
    }

    #[test]
    fn reduce_identity() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(reduce_to_shape(&data, &[3], &[3]), data);
    }
}
