#[derive(Debug, Default)]
pub struct Tensor<T> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
}