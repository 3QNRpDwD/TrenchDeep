use super::*;

impl Function for Add {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let first_shape = first_target.shape();
        let second_shape = second_target.shape();

        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_target.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_target.data()[i * features + j] + second_target.data()[j];
                }
            }
            
            return Ok(vec![PooledTensor::from_vec(data, first_shape).unwrap()])
        }

        match first_target.chk_shape(second_target) {
            Err(e) => Err(e),
            _ => Ok(vec![PooledTensor::from_vec(self.backend().add(first_target.data(), second_target.data()), first_target.shape()).unwrap()])
        }
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: HandleId) -> MlResult<Vec<Tensor>> {
        let first_target = targets[0];
        let second_target = targets[1];
        let first_shape = first_target.shape();
        let second_shape = second_target.shape();

        if first_shape.len() == 2 && second_shape.len() == 1 && first_shape[1] == second_shape[0] {
            // Special case for matrix + vector broadcasting
            let (batch_size, features) = (first_shape[0], first_shape[1]);
            let mut data = vec![0.0; first_target.data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = first_target.data()[i * features + j] + second_target.data()[j];
                }
            }

            return Ok(vec![Tensor::with_id(data, first_shape, node_id).unwrap()])
        }

        match first_target.chk_shape(second_target) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::with_id(self.backend().add(first_target.data(), second_target.data()), first_target.shape(), node_id).unwrap()])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let gt = PooledTensor::from_vec(grad.data().to_vec(), grad.shape()).unwrap();
        Ok(vec![gt.clone(), gt])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }
    
    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = PooledTensor;

    fn add(self, other: Tensor) -> Self::Output {
        let op = Add::new();
        op.forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = PooledTensor;

    fn add(self, other: &Tensor) -> Self::Output {
        let op = Add::new();
        op.forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = PooledTensor;

    fn add(self, other: &Tensor) -> Self::Output {
        let op = Add::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Add<Tensor> for &Tensor {
    type Output = PooledTensor;

    fn add(self, other: Tensor) -> Self::Output {
        let op = Add::new();
        op.forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Add<&dyn TensorBase> for &dyn TensorBase {
    type Output = PooledTensor;

    fn add(self, other: &dyn TensorBase) -> Self::Output {
        let op = Add::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, other: Tensor) {
        let op = Add::new();
        op.assign_forward(&[self, &other], self.0).unwrap();
    }
}

impl std::ops::AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, other: &Tensor) {
        let op = Add::new();
        op.assign_forward(&[self, other], self.0).unwrap();
    }
}
