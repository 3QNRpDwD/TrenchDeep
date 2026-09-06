use super::*;
impl ExecutionContext {
    pub fn add(&self, input: &Tensor, rhs: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Add, &[input, rhs])
    }
}
impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> MlResult<Self> {
        self.execution_context()?.add(self, rhs)
    }
}
impl Variable {
    pub fn add(&self, rhs: &Tensor) -> MlResult<Self> {
        self.tensor.add(rhs)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn sub(&self, input: &Tensor, rhs: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Sub, &[input, rhs])
    }
}
impl Tensor {
    pub fn sub(&self, rhs: &Tensor) -> MlResult<Self> {
        self.execution_context()?.sub(self, rhs)
    }
}
impl Variable {
    pub fn sub(&self, rhs: &Tensor) -> MlResult<Self> {
        self.tensor.sub(rhs)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn mul(&self, input: &Tensor, rhs: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Mul, &[input, rhs])
    }
}
impl Tensor {
    pub fn mul(&self, rhs: &Tensor) -> MlResult<Self> {
        self.execution_context()?.mul(self, rhs)
    }
}
impl Variable {
    pub fn mul(&self, rhs: &Tensor) -> MlResult<Self> {
        self.tensor.mul(rhs)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn div(&self, input: &Tensor, rhs: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Div, &[input, rhs])
    }
}
impl Tensor {
    pub fn div(&self, rhs: &Tensor) -> MlResult<Self> {
        self.execution_context()?.div(self, rhs)
    }
}
impl Variable {
    pub fn div(&self, rhs: &Tensor) -> MlResult<Self> {
        self.tensor.div(rhs)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn matmul(&self, input: &Tensor, rhs: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Matmul, &[input, rhs])
    }
}
impl Tensor {
    pub fn matmul(&self, rhs: &Tensor) -> MlResult<Self> {
        self.execution_context()?.matmul(self, rhs)
    }
}
impl Variable {
    pub fn matmul(&self, rhs: &Tensor) -> MlResult<Self> {
        self.tensor.matmul(rhs)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn neg(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Neg, &[input])
    }
}
impl Tensor {
    pub fn neg(&self) -> MlResult<Self> {
        self.execution_context()?.neg(self)
    }
}
impl Variable {
    pub fn neg(&self) -> MlResult<Self> {
        self.tensor.neg()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn square(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Square, &[input])
    }
}
impl Tensor {
    pub fn square(&self) -> MlResult<Self> {
        self.execution_context()?.square(self)
    }
}
impl Variable {
    pub fn square(&self) -> MlResult<Self> {
        self.tensor.square()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn exp(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Exp, &[input])
    }
}
impl Tensor {
    pub fn exp(&self) -> MlResult<Self> {
        self.execution_context()?.exp(self)
    }
}
impl Variable {
    pub fn exp(&self) -> MlResult<Self> {
        self.tensor.exp()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn log(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Log, &[input])
    }
}
impl Tensor {
    pub fn log(&self) -> MlResult<Self> {
        self.execution_context()?.log(self)
    }
}
impl Variable {
    pub fn log(&self) -> MlResult<Self> {
        self.tensor.log()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn sqrt(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Sqrt, &[input])
    }
}
impl Tensor {
    pub fn sqrt(&self) -> MlResult<Self> {
        self.execution_context()?.sqrt(self)
    }
}
impl Variable {
    pub fn sqrt(&self) -> MlResult<Self> {
        self.tensor.sqrt()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn abs(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Abs, &[input])
    }
}
impl Tensor {
    pub fn abs(&self) -> MlResult<Self> {
        self.execution_context()?.abs(self)
    }
}
impl Variable {
    pub fn abs(&self) -> MlResult<Self> {
        self.tensor.abs()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn sin(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Sin, &[input])
    }
}
impl Tensor {
    pub fn sin(&self) -> MlResult<Self> {
        self.execution_context()?.sin(self)
    }
}
impl Variable {
    pub fn sin(&self) -> MlResult<Self> {
        self.tensor.sin()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn cos(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Cos, &[input])
    }
}
impl Tensor {
    pub fn cos(&self) -> MlResult<Self> {
        self.execution_context()?.cos(self)
    }
}
impl Variable {
    pub fn cos(&self) -> MlResult<Self> {
        self.tensor.cos()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn tanh(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Tanh, &[input])
    }
}
impl Tensor {
    pub fn tanh(&self) -> MlResult<Self> {
        self.execution_context()?.tanh(self)
    }
}
impl Variable {
    pub fn tanh(&self) -> MlResult<Self> {
        self.tensor.tanh()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn sigmoid(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Sigmoid, &[input])
    }
}
impl Tensor {
    pub fn sigmoid(&self) -> MlResult<Self> {
        self.execution_context()?.sigmoid(self)
    }
}
impl Variable {
    pub fn sigmoid(&self) -> MlResult<Self> {
        self.tensor.sigmoid()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn silu(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Silu, &[input])
    }
}
impl Tensor {
    pub fn silu(&self) -> MlResult<Self> {
        self.execution_context()?.silu(self)
    }
}
impl Variable {
    pub fn silu(&self) -> MlResult<Self> {
        self.tensor.silu()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn relu(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Relu, &[input])
    }
}
impl Tensor {
    pub fn relu(&self) -> MlResult<Self> {
        self.execution_context()?.relu(self)
    }
}
impl Variable {
    pub fn relu(&self) -> MlResult<Self> {
        self.tensor.relu()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn sum(&self, input: &Tensor) -> MlResult<Tensor> {
        self.single(Operation::Sum, &[input])
    }
}
impl Tensor {
    pub fn sum(&self) -> MlResult<Self> {
        self.execution_context()?.sum(self)
    }
}
impl Variable {
    pub fn sum(&self) -> MlResult<Self> {
        self.tensor.sum()?.as_variable()
    }
}
impl ExecutionContext {
    pub fn powf(&self, input: &Tensor, exponent: f32) -> MlResult<Tensor> {
        self.single(Operation::Pow(exponent), &[input])
    }
}
impl Tensor {
    pub fn powf(&self, exponent: f32) -> MlResult<Self> {
        self.execution_context()?.powf(self, exponent)
    }
}
impl Variable {
    pub fn powf(&self, exponent: f32) -> MlResult<Self> {
        self.tensor.powf(exponent)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn reshape(&self, input: &Tensor, shape: &[usize]) -> MlResult<Tensor> {
        self.single(Operation::Reshape(shape.to_vec()), &[input])
    }
}
impl Tensor {
    pub fn reshape(&self, shape: &[usize]) -> MlResult<Self> {
        self.execution_context()?.reshape(self, shape)
    }
}
impl Variable {
    pub fn reshape(&self, shape: &[usize]) -> MlResult<Self> {
        self.tensor.reshape(shape)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn transpose(&self, input: &Tensor, axes: &[usize]) -> MlResult<Tensor> {
        self.single(Operation::Transpose(axes.to_vec()), &[input])
    }
}
impl Tensor {
    pub fn transpose(&self, axes: &[usize]) -> MlResult<Self> {
        self.execution_context()?.transpose(self, axes)
    }
}
impl Variable {
    pub fn transpose(&self, axes: &[usize]) -> MlResult<Self> {
        self.tensor.transpose(axes)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn softmax(&self, input: &Tensor, axis: usize) -> MlResult<Tensor> {
        self.single(Operation::Softmax { axis }, &[input])
    }
}
impl Tensor {
    pub fn softmax(&self, axis: usize) -> MlResult<Self> {
        self.execution_context()?.softmax(self, axis)
    }
}
impl Variable {
    pub fn softmax(&self, axis: usize) -> MlResult<Self> {
        self.tensor.softmax(axis)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn approx_sin(&self, input: &Tensor, threshold: f32) -> MlResult<Tensor> {
        self.single(Operation::ApproxSin { threshold }, &[input])
    }
}
impl Tensor {
    pub fn approx_sin(&self, threshold: f32) -> MlResult<Self> {
        self.execution_context()?.approx_sin(self, threshold)
    }
}
impl Variable {
    pub fn approx_sin(&self, threshold: f32) -> MlResult<Self> {
        self.tensor.approx_sin(threshold)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn approx_cos(&self, input: &Tensor, threshold: f32) -> MlResult<Tensor> {
        self.single(Operation::ApproxCos { threshold }, &[input])
    }
}
impl Tensor {
    pub fn approx_cos(&self, threshold: f32) -> MlResult<Self> {
        self.execution_context()?.approx_cos(self, threshold)
    }
}
impl Variable {
    pub fn approx_cos(&self, threshold: f32) -> MlResult<Self> {
        self.tensor.approx_cos(threshold)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn conv2d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Conv2d { stride, padding },
            &[input, weight, bias],
        )
    }
}
impl Tensor {
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<Self> {
        self.execution_context()?
            .conv2d(self, weight, bias, stride, padding)
    }
}
impl Variable {
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<Self> {
        self.tensor
            .conv2d(weight, bias, stride, padding)?
            .as_variable()
    }
}
impl ExecutionContext {
    pub fn max_pool2d(
        &self,
        input: &Tensor,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<Tensor> {
        self.single(Operation::MaxPool2d { kernel, stride }, &[input])
    }
}
impl Tensor {
    pub fn max_pool2d(&self, kernel: (usize, usize), stride: (usize, usize)) -> MlResult<Self> {
        self.execution_context()?.max_pool2d(self, kernel, stride)
    }
}
impl Variable {
    pub fn max_pool2d(&self, kernel: (usize, usize), stride: (usize, usize)) -> MlResult<Self> {
        self.tensor.max_pool2d(kernel, stride)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn avg_pool2d(
        &self,
        input: &Tensor,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<Tensor> {
        self.single(Operation::AvgPool2d { kernel, stride }, &[input])
    }
}
impl Tensor {
    pub fn avg_pool2d(&self, kernel: (usize, usize), stride: (usize, usize)) -> MlResult<Self> {
        self.execution_context()?.avg_pool2d(self, kernel, stride)
    }
}
impl Variable {
    pub fn avg_pool2d(&self, kernel: (usize, usize), stride: (usize, usize)) -> MlResult<Self> {
        self.tensor.avg_pool2d(kernel, stride)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn nearest_upsample2d(&self, input: &Tensor, scale: (usize, usize)) -> MlResult<Tensor> {
        self.single(Operation::NearestUpsample2d { scale }, &[input])
    }
}
impl Tensor {
    pub fn nearest_upsample2d(&self, scale: (usize, usize)) -> MlResult<Self> {
        self.execution_context()?.nearest_upsample2d(self, scale)
    }
}
impl Variable {
    pub fn nearest_upsample2d(&self, scale: (usize, usize)) -> MlResult<Self> {
        self.tensor.nearest_upsample2d(scale)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn group_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        groups: usize,
        epsilon: f32,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::GroupNorm { groups, epsilon },
            &[input, gamma, beta],
        )
    }
}
impl Tensor {
    pub fn group_norm(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        groups: usize,
        epsilon: f32,
    ) -> MlResult<Self> {
        self.execution_context()?
            .group_norm(self, gamma, beta, groups, epsilon)
    }
}
impl Variable {
    pub fn group_norm(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        groups: usize,
        epsilon: f32,
    ) -> MlResult<Self> {
        self.tensor
            .group_norm(gamma, beta, groups, epsilon)?
            .as_variable()
    }
}
impl ExecutionContext {
    pub fn mse_loss(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Loss {
                kind: LossKind::Mse,
                reduction,
            },
            &[input, target],
        )
    }
}
impl Tensor {
    pub fn mse_loss(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.execution_context()?.mse_loss(self, target, reduction)
    }
}
impl Variable {
    pub fn mse_loss(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.tensor.mse_loss(target, reduction)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn mae_loss(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Loss {
                kind: LossKind::Mae,
                reduction,
            },
            &[input, target],
        )
    }
}
impl Tensor {
    pub fn mae_loss(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.execution_context()?.mae_loss(self, target, reduction)
    }
}
impl Variable {
    pub fn mae_loss(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.tensor.mae_loss(target, reduction)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn huber_loss(
        &self,
        input: &Tensor,
        target: &Tensor,
        delta: f32,
        reduction: Reduction,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Loss {
                kind: LossKind::Huber { delta },
                reduction,
            },
            &[input, target],
        )
    }
}
impl Tensor {
    pub fn huber_loss(&self, target: &Tensor, delta: f32, reduction: Reduction) -> MlResult<Self> {
        self.execution_context()?
            .huber_loss(self, target, delta, reduction)
    }
}
impl Variable {
    pub fn huber_loss(&self, target: &Tensor, delta: f32, reduction: Reduction) -> MlResult<Self> {
        self.tensor
            .huber_loss(target, delta, reduction)?
            .as_variable()
    }
}
impl ExecutionContext {
    pub fn binary_cross_entropy(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Loss {
                kind: LossKind::BinaryCrossEntropy,
                reduction,
            },
            &[input, target],
        )
    }
}
impl Tensor {
    pub fn binary_cross_entropy(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.execution_context()?
            .binary_cross_entropy(self, target, reduction)
    }
}
impl Variable {
    pub fn binary_cross_entropy(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.tensor
            .binary_cross_entropy(target, reduction)?
            .as_variable()
    }
}
impl ExecutionContext {
    pub fn cross_entropy(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Loss {
                kind: LossKind::CrossEntropy,
                reduction,
            },
            &[input, target],
        )
    }
}
impl Tensor {
    pub fn cross_entropy(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.execution_context()?
            .cross_entropy(self, target, reduction)
    }
}
impl Variable {
    pub fn cross_entropy(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.tensor.cross_entropy(target, reduction)?.as_variable()
    }
}
impl ExecutionContext {
    pub fn softmax_cross_entropy(
        &self,
        input: &Tensor,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Tensor> {
        self.single(
            Operation::Loss {
                kind: LossKind::SoftmaxCrossEntropy,
                reduction,
            },
            &[input, target],
        )
    }
}
impl Tensor {
    pub fn softmax_cross_entropy(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.execution_context()?
            .softmax_cross_entropy(self, target, reduction)
    }
}
impl Variable {
    pub fn softmax_cross_entropy(&self, target: &Tensor, reduction: Reduction) -> MlResult<Self> {
        self.tensor
            .softmax_cross_entropy(target, reduction)?
            .as_variable()
    }
}
impl ExecutionContext {
    pub fn concat(&self, inputs: &[&Tensor], axis: usize) -> MlResult<Tensor> {
        self.single(Operation::Concat { axis }, inputs)
    }
    pub fn topk(&self, input: &Tensor, k: usize, sorted: bool) -> MlResult<TopKResult> {
        let mut outputs = self
            .execute(&Operation::TopK { k, sorted }, &[input])?
            .into_iter();
        Ok(TopKResult {
            values: outputs.next().ok_or(TensorError::EmptyTensor)?,
            indices: outputs.next().ok_or(TensorError::EmptyTensor)?,
        })
    }
    pub fn matmax(
        &self,
        input: &Tensor,
        axis: Option<isize>,
        keepdim: bool,
    ) -> MlResult<MaxResult> {
        let mut outputs = self
            .execute(&Operation::Matmax { axis, keepdim }, &[input])?
            .into_iter();
        Ok(MaxResult {
            values: outputs.next().ok_or(TensorError::EmptyTensor)?,
            indices: outputs.next().ok_or(TensorError::EmptyTensor)?,
        })
    }
}
impl Tensor {
    pub fn topk(&self, k: usize, sorted: bool) -> MlResult<TopKResult> {
        self.execution_context()?.topk(self, k, sorted)
    }
    pub fn matmax(&self, axis: Option<isize>, keepdim: bool) -> MlResult<MaxResult> {
        self.execution_context()?.matmax(self, axis, keepdim)
    }
}
impl std::ops::Add<&Tensor> for &Tensor {
    type Output = MlResult<Tensor>;
    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs)
    }
}
impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = MlResult<Tensor>;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = MlResult<Tensor>;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}
impl std::ops::Div<&Tensor> for &Tensor {
    type Output = MlResult<Tensor>;
    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(self, rhs)
    }
}
impl std::ops::Neg for &Tensor {
    type Output = MlResult<Tensor>;
    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}
impl std::ops::Add<&Variable> for &Variable {
    type Output = MlResult<Variable>;
    fn add(self, rhs: &Variable) -> Self::Output {
        Variable::add(self, rhs.tensor())
    }
}
impl std::ops::Sub<&Variable> for &Variable {
    type Output = MlResult<Variable>;
    fn sub(self, rhs: &Variable) -> Self::Output {
        Variable::sub(self, rhs.tensor())
    }
}
impl std::ops::Mul<&Variable> for &Variable {
    type Output = MlResult<Variable>;
    fn mul(self, rhs: &Variable) -> Self::Output {
        Variable::mul(self, rhs.tensor())
    }
}
impl std::ops::Div<&Variable> for &Variable {
    type Output = MlResult<Variable>;
    fn div(self, rhs: &Variable) -> Self::Output {
        Variable::div(self, rhs.tensor())
    }
}
impl std::ops::Neg for &Variable {
    type Output = MlResult<Variable>;
    fn neg(self) -> Self::Output {
        Variable::neg(self)
    }
}
impl ExecutionContext {
    pub(crate) fn add_variable(&self, lhs: &Variable, rhs: &Variable) -> MlResult<Variable> {
        let tensor = self.add(lhs.tensor(), rhs.tensor())?;
        Ok(Variable { tensor })
    }

    pub(crate) fn mul_variable(&self, lhs: &Variable, rhs: &Variable) -> MlResult<Variable> {
        let tensor = self.mul(lhs.tensor(), rhs.tensor())?;
        Ok(Variable { tensor })
    }

    pub(crate) fn variable_from(&self, tensor: Tensor) -> MlResult<Variable> {
        self.validate(&tensor)?;
        Ok(Variable { tensor })
    }

    pub(crate) fn sub_variable(&self, lhs: &Variable, rhs: &Variable) -> MlResult<Variable> {
        self.variable_from(self.sub(lhs.tensor(), rhs.tensor())?)
    }

    pub(crate) fn div_variable(&self, lhs: &Variable, rhs: &Variable) -> MlResult<Variable> {
        self.variable_from(self.div(lhs.tensor(), rhs.tensor())?)
    }

    pub(crate) fn square_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.square(input.tensor())?)
    }

    pub(crate) fn neg_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.neg(input.tensor())?)
    }

    pub(crate) fn exp_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.exp(input.tensor())?)
    }

    pub(crate) fn log_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.log(input.tensor())?)
    }

    pub(crate) fn sqrt_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.sqrt(input.tensor())?)
    }

    pub(crate) fn powf_variable(&self, input: &Variable, exponent: f32) -> MlResult<Variable> {
        self.variable_from(self.powf(input.tensor(), exponent)?)
    }

    pub(crate) fn sin_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.sin(input.tensor())?)
    }

    pub(crate) fn cos_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.cos(input.tensor())?)
    }

    pub(crate) fn approx_sin_variable(
        &self,
        input: &Variable,
        threshold: f32,
    ) -> MlResult<Variable> {
        self.variable_from(self.approx_sin(input.tensor(), threshold)?)
    }

    pub(crate) fn approx_cos_variable(
        &self,
        input: &Variable,
        threshold: f32,
    ) -> MlResult<Variable> {
        self.variable_from(self.approx_cos(input.tensor(), threshold)?)
    }

    pub(crate) fn tanh_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.tanh(input.tensor())?)
    }

    pub(crate) fn sigmoid_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.sigmoid(input.tensor())?)
    }

    pub(crate) fn silu_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.silu(input.tensor())?)
    }

    pub(crate) fn relu_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.relu(input.tensor())?)
    }

    pub(crate) fn abs_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.abs(input.tensor())?)
    }

    pub(crate) fn softmax_variable(&self, input: &Variable, axis: usize) -> MlResult<Variable> {
        self.variable_from(self.softmax(input.tensor(), axis)?)
    }

    pub(crate) fn sum_variable(&self, input: &Variable) -> MlResult<Variable> {
        self.variable_from(self.sum(input.tensor())?)
    }

    pub(crate) fn mse_loss_variable(
        &self,
        prediction: &Variable,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Variable> {
        self.variable_from(self.mse_loss(prediction.tensor(), target, reduction)?)
    }

    pub(crate) fn mae_loss_variable(
        &self,
        prediction: &Variable,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Variable> {
        self.variable_from(self.mae_loss(prediction.tensor(), target, reduction)?)
    }

    pub(crate) fn huber_loss_variable(
        &self,
        prediction: &Variable,
        target: &Tensor,
        delta: f32,
        reduction: Reduction,
    ) -> MlResult<Variable> {
        self.variable_from(self.huber_loss(prediction.tensor(), target, delta, reduction)?)
    }

    pub(crate) fn binary_cross_entropy_variable(
        &self,
        prediction: &Variable,
        target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Variable> {
        self.variable_from(self.binary_cross_entropy(prediction.tensor(), target, reduction)?)
    }

    pub(crate) fn cross_entropy_variable(
        &self,
        probabilities: &Variable,
        one_hot_target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Variable> {
        self.variable_from(self.cross_entropy(probabilities.tensor(), one_hot_target, reduction)?)
    }

    pub(crate) fn softmax_cross_entropy_variable(
        &self,
        logits: &Variable,
        one_hot_target: &Tensor,
        reduction: Reduction,
    ) -> MlResult<Variable> {
        self.variable_from(self.softmax_cross_entropy(
            logits.tensor(),
            one_hot_target,
            reduction,
        )?)
    }

    pub(crate) fn reshape_variable(&self, input: &Variable, shape: &[usize]) -> MlResult<Variable> {
        self.variable_from(self.reshape(input.tensor(), shape)?)
    }

    pub(crate) fn transpose_variable(
        &self,
        input: &Variable,
        axes: &[usize],
    ) -> MlResult<Variable> {
        self.variable_from(self.transpose(input.tensor(), axes)?)
    }

    pub(crate) fn concat_variables(&self, inputs: &[&Variable], axis: usize) -> MlResult<Variable> {
        let tensors: Vec<_> = inputs.iter().map(|input| input.tensor()).collect();
        self.variable_from(self.concat(&tensors, axis)?)
    }

    pub(crate) fn matmul_variable(&self, lhs: &Variable, rhs: &Variable) -> MlResult<Variable> {
        self.variable_from(self.matmul(lhs.tensor(), rhs.tensor())?)
    }

    pub(crate) fn conv2d_variable(
        &self,
        input: &Variable,
        weight: &Variable,
        bias: &Variable,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MlResult<Variable> {
        self.variable_from(self.conv2d(
            input.tensor(),
            weight.tensor(),
            bias.tensor(),
            stride,
            padding,
        )?)
    }

    pub(crate) fn max_pool2d_variable(
        &self,
        input: &Variable,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<Variable> {
        self.variable_from(self.max_pool2d(input.tensor(), kernel, stride)?)
    }

    pub(crate) fn avg_pool2d_variable(
        &self,
        input: &Variable,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MlResult<Variable> {
        self.variable_from(self.avg_pool2d(input.tensor(), kernel, stride)?)
    }

    pub(crate) fn nearest_upsample2d_variable(
        &self,
        input: &Variable,
        scale: (usize, usize),
    ) -> MlResult<Variable> {
        self.variable_from(self.nearest_upsample2d(input.tensor(), scale)?)
    }

    pub(crate) fn group_norm_variable(
        &self,
        input: &Variable,
        gamma: &Variable,
        beta: &Variable,
        groups: usize,
        epsilon: f32,
    ) -> MlResult<Variable> {
        self.variable_from(self.group_norm(
            input.tensor(),
            gamma.tensor(),
            beta.tensor(),
            groups,
            epsilon,
        )?)
    }
}
