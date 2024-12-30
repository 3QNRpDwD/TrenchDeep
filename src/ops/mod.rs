mod arithmetic;
mod traits;

pub use traits::TensorOps;

#[cfg(test)]
mod tests {
    use crate::core::{Tensor, TensorLayer};
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

    #[test]
    fn tensor_arithmetic_operations() {
        let t1: Tensor<Vec<f32>> = TensorLayer::new(vec![1.0, 2.0]);
        let t2: Tensor<Vec<f32>> = TensorLayer::new(vec![3.0, 4.0]);

        assert_tensor_eq(t1.add(&t2), vec![6.0, 6.0, 6.0, 6.0], vec![2, 2]);
        assert_tensor_eq(t1.sub(&t2), vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
        assert_tensor_eq(t1.mul(&t2), vec![8.0, 8.0, 8.0, 8.0], vec![2, 2]);
        assert_tensor_eq(t1.div(&t2), vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
    }
}