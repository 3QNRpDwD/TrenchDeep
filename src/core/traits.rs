pub trait TensorLayer<T> {
    fn new(shape: Vec<usize>, value: T) -> Self where T: Clone;
    fn get(&self, indices: &[usize]) -> Option<&T>;
    fn index(&self, indices: &[usize]) -> Option<usize>;
}