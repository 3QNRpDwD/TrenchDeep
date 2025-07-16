use super::*;

impl Function for Sub {
    fn forward(&self, targets: &[&dyn TensorBase]) -> MlResult<Vec<PooledTensor>> {
        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![PooledTensor::from_vec(data, &targets[0].shape()).unwrap()])
        }

        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![PooledTensor::from_vec(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape()).unwrap()])
        }
    }

    fn assign_forward(&self, targets: &[&dyn TensorBase], node_id: HandleId) -> MlResult<Vec<Tensor>> {
        if targets[0].shape().len() == 2 && targets[1].shape().len() == 1 && targets[0].shape()[1] == targets[1].shape()[0] {
            let (batch_size, features) = (targets[0].shape()[0], targets[0].shape()[1]);
            let mut data = vec![0.0; targets[0].data().len()];

            for i in 0..batch_size {
                for j in 0..features {
                    data[i * features + j] = targets[0].data()[i * features + j] - targets[1].data()[j];
                }
            }
            return Ok(vec![Tensor::with_id(data, &targets[0].shape(), node_id).unwrap()])
        }

        match targets[0].chk_shape(targets[1]) {
            Err(e) => Err(e),
            _ => Ok(vec![Tensor::with_id(self.backend().sub(targets[0].data(), targets[1].data()), targets[0].shape(), node_id).unwrap()])
        }
    }

    #[cfg(all(feature = "enableBackpropagation"))]
    fn backward(&self, _: &[&dyn TensorBase], grad: &dyn TensorBase) -> MlResult<Vec<PooledTensor>> {
        let gt = PooledTensor::from_vec(grad.data().to_vec(), grad.shape()).unwrap();
        Ok(vec![gt, PooledTensor::from_vec(grad.data().iter().map(|&x| -x).collect(), grad.shape()).unwrap()])
    }

    fn backend(&self) -> &Arc<dyn Backend> { &self.backend }

    fn node_id(&self) -> &HandleId { &self.node_id }
}

impl std::ops::Sub<Tensor> for Tensor {
    type Output = PooledTensor;

    fn sub(self, other: Tensor) -> Self::Output {
        let op = Sub::new();
        op.forward(&[&self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor> for Tensor {
    type Output = PooledTensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        let op = Sub::new();
        op.forward(&[&self, other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = PooledTensor;

    fn sub(self, other: &Tensor) -> Self::Output {
        let op = Sub::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<Tensor> for &Tensor {
    type Output = PooledTensor;

    fn sub(self, other: Tensor) -> Self::Output {
        let op = Sub::new();
        op.forward(&[self, &other]).unwrap().remove(0)
    }
}

impl std::ops::Sub<&dyn TensorBase> for &dyn TensorBase {
    type Output = PooledTensor;

    fn sub(self, other: &dyn TensorBase) -> Self::Output {
        let op = Sub::new();
        op.forward(&[self, other]).unwrap().remove(0)
    }
}

impl std::ops::SubAssign<Tensor> for Tensor {
    fn sub_assign(&mut self, other: Tensor) {
        let op = Sub::new();
        op.assign_forward(&[self, &other], self.0).unwrap();
    }
}

impl std::ops::SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, other: &Tensor) {
        let op = Sub::new();
        op.assign_forward(&[self, other], self.0).unwrap();
    }
}
