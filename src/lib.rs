mod core;
mod ops;
mod broadcast;
mod test;

pub use core::Tensor;
pub use core::TensorLayer;
pub use ops::TensorArithmetic;

#[cfg(test)]
mod tests {
    use super::*;

    // Option<T>의 결과를 테스트하는 헬퍼 함수
    pub fn assert_tensor_eq<T: PartialEq + std::fmt::Debug>(
        result: Option<Tensor<T>>,
        expected_data: Vec<T>,
        expected_shape: Vec<usize>
    ) {
        let tensor = result.unwrap();
        debug_assert_eq!(tensor.data, expected_data);
        debug_assert_eq!(tensor.shape, expected_shape);
    }

    // 헬퍼 함수를 사용한 테스트 예시
    #[test]
    pub fn tensor_add_with_helper_test() {
        let t1: Tensor<f32> = TensorLayer::new(vec![2, 2], 1.0);
        let t2: Tensor<f32> = TensorLayer::new(vec![2, 2], 2.0);

        assert_tensor_eq(
            t1.add(&t2),
            vec![3.0, 3.0, 3.0, 3.0],
            vec![2, 2]
        );
    }

    #[test]
    fn tensor_arithmetic_operations() {
        let t1: Tensor<f32> = TensorLayer::new(vec![2, 2], 4.0);
        let t2: Tensor<f32> = TensorLayer::new(vec![2, 2], 2.0);

        assert_tensor_eq(t1.add(&t2), vec![6.0, 6.0, 6.0, 6.0], vec![2, 2]);
        assert_tensor_eq(t1.sub(&t2), vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
        assert_tensor_eq(t1.mul(&t2), vec![8.0, 8.0, 8.0, 8.0], vec![2, 2]);
        assert_tensor_eq(t1.div(&t2), vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
    }
}