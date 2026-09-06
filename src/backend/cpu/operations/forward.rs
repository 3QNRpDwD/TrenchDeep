use super::*;

use crate::backend::CpuBackend;
impl OperationProvider for CpuBackend {
    fn execute(&self, operation: &Operation, inputs: &[TensorView<'_>]) -> MlResult<OperationOutput> {
        if inputs.is_empty() || operation.input_count().is_some_and(|n| n != inputs.len()) {
            return Err(TensorError::InvalidOperation { op: operation.name(), reason: "invalid input count".into() }.into());
        }
        let values = inputs.iter().map(|x| (*x).to_owned()).collect::<MlResult<Vec<_>>>()?;
        let x = &values[0];
        let mut saved = Vec::new();
        let (output, backward) = match operation {
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div => {
                let y = &values[1];
                let shape = broadcast_shape(&x.shape, &y.shape).ok_or_else(|| TensorError::InvalidOperation {
                    op: operation.name(), reason: "incompatible broadcast shapes".into(),
                })?;
                let a = broadcast_data(x, &shape)?;
                let b = broadcast_data(y, &shape)?;
                let data = match operation {
                    Operation::Add => self.compute.add(&a,&b), Operation::Sub => self.compute.sub(&a,&b),
                    Operation::Mul => self.compute.multiply(&a,&b), _ => self.compute.div(&a,&b),
                };
                let backward = match operation { Operation::Add => BuiltinBackward::Add, Operation::Sub => BuiltinBackward::Sub,
                    Operation::Mul => BuiltinBackward::Mul, _ => BuiltinBackward::Div };
                (TensorBuffer::from_vec(data, &shape)?, backward)
            }
            Operation::Neg => (tensor_map(x, |x| -x)?, BuiltinBackward::Neg),
            Operation::Square => (tensor_map(x, |x| x*x)?, BuiltinBackward::Square),
            Operation::Exp => (TensorBuffer::from_vec(self.compute.exp(&x.data),&x.shape)?, BuiltinBackward::Exp),
            Operation::Log => (TensorBuffer::from_vec(self.compute.log(&x.data),&x.shape)?, BuiltinBackward::Log),
            Operation::Sqrt => (TensorBuffer::from_vec(self.compute.sqrt(&x.data),&x.shape)?, BuiltinBackward::Sqrt),
            Operation::Pow(p) => (TensorBuffer::from_vec(self.compute.pow(&x.data,*p),&x.shape)?, BuiltinBackward::Pow(*p)),
            Operation::Sin => (tensor_map(x, f32::sin)?, BuiltinBackward::Sin),
            Operation::Cos => (tensor_map(x, f32::cos)?, BuiltinBackward::Cos),
            Operation::ApproxSin { threshold } => {
                validate_approx_threshold(operation.name(), *threshold)?;
                (tensor_map(x, approx_sin_value)?, BuiltinBackward::ApproxSin { threshold: *threshold })
            }
            Operation::ApproxCos { threshold } => {
                validate_approx_threshold(operation.name(), *threshold)?;
                (tensor_map(x, approx_cos_value)?, BuiltinBackward::ApproxCos { threshold: *threshold })
            }
            Operation::Tanh => (tensor_map(x, f32::tanh)?, BuiltinBackward::Tanh),
            Operation::Sigmoid => (tensor_map(x, |x| 1.0/(1.0+(-x).exp()))?, BuiltinBackward::Sigmoid),
            Operation::Silu => (tensor_map(x, |x| x/(1.0+(-x).exp()))?, BuiltinBackward::Silu),
            Operation::Relu => (tensor_map(x, |x| x.max(0.0))?, BuiltinBackward::Relu),
            Operation::Abs => (tensor_map(x, f32::abs)?, BuiltinBackward::Abs),
            Operation::Softmax { axis } => {
                let axis = *axis;
                if axis >= x.shape.len() { return Err(TensorError::InvalidAxis { axis, shape: x.shape.clone() }.into()); }
                let outer: usize = x.shape[..axis].iter().product();
                let width = x.shape[axis];
                if width == 0 { return Err(TensorError::EmptyTensor.into()); }
                let inner: usize = x.shape[axis+1..].iter().product();
                let mut data = vec![0.0; x.data.len()];
                for o in 0..outer { for i in 0..inner {
                    let max = (0..width).map(|j| x.data[(o*width+j)*inner+i]).fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = (0..width).map(|j| (x.data[(o*width+j)*inner+i]-max).exp()).sum();
                    for j in 0..width { let idx = (o*width+j)*inner+i; data[idx] = (x.data[idx]-max).exp()/sum; }
                }}
                (TensorBuffer::from_vec(data, &x.shape)?, BuiltinBackward::Softmax { axis })
            }
            Operation::Sum => (TensorBuffer::from_vec(vec![self.compute.sum(&x.data)], &[])?, BuiltinBackward::Sum),
            Operation::Reshape(shape) => (TensorBuffer::from_vec(x.data.clone(), shape)?, BuiltinBackward::Reshape),
            Operation::Transpose(axes) => {
                validate_permutation(&x.shape, axes)?;
                let shape = axes.iter().map(|&i| x.shape[i]).collect::<Vec<_>>();
                (TensorBuffer::from_vec(permute_data(&x.data, &x.shape, axes), &shape)?, BuiltinBackward::Transpose(axes.clone()))
            }
            Operation::Concat { axis } => {
                let axis = *axis;
                if axis >= x.shape.len() { return Err(TensorError::InvalidAxis { axis, shape: x.shape.clone() }.into()); }
                for value in &values {
                    if value.shape.len() != x.shape.len() || value.shape.iter().enumerate().any(|(i,&d)| i != axis && d != x.shape[i]) {
                        return Err(TensorError::InvalidOperation { op: "concat", reason: "non-concatenated dimensions must match".into() }.into());
                    }
                }
                let mut shape = x.shape.clone();
                shape[axis] = values.iter().try_fold(0usize, |n,v| n.checked_add(v.shape[axis]))
                    .ok_or_else(|| TensorError::InvalidOperation { op: "concat", reason: "shape overflow".into() })?;
                let outer: usize = shape[..axis].iter().product();
                let inner: usize = shape[axis+1..].iter().product();
                let mut data = Vec::new();
                for o in 0..outer { for value in &values {
                    let chunk = value.shape[axis]*inner;
                    data.extend_from_slice(&value.data[o*chunk..(o+1)*chunk]);
                }}
                (TensorBuffer::from_vec(data, &shape)?, BuiltinBackward::Concat { axis, sizes: values.iter().map(|v| v.shape[axis]).collect() })
            }
            Operation::Matmul => {
                let y = &values[1];
                let spec = MatmulSpec::new(&x.shape, &y.shape)?;
                let batches: usize = spec.batch_shape.iter().product();
                let mut data = vec![0.0; batches*spec.m*spec.n];
                for b in 0..batches {
                    let lb = broadcast_offset(b, &spec.batch_shape, &spec.left_batch);
                    let rb = broadcast_offset(b, &spec.batch_shape, &spec.right_batch);
                    let left=&x.data[lb*spec.m*spec.k..(lb+1)*spec.m*spec.k];
                    let right=&y.data[rb*spec.k*spec.n..(rb+1)*spec.k*spec.n];
                    let product=self.compute.matmul(left,right,spec.m,spec.k,spec.n);
                    data[b*spec.m*spec.n..(b+1)*spec.m*spec.n].copy_from_slice(&product);
                }
                (TensorBuffer::from_vec(data, &spec.output_shape)?, BuiltinBackward::Matmul)
            }
            Operation::Conv2d {stride,padding} => (conv2d_forward_data(x, &values[1], &values[2], *stride, *padding)?, BuiltinBackward::Conv2d {stride:*stride,padding:*padding}),
            Operation::MaxPool2d {kernel,stride} => {
                let (out,mask) = max_pool2d_forward_data(x,*kernel,*stride)?;
                saved.push(mask); (out,BuiltinBackward::MaxPool2d {kernel:*kernel,stride:*stride})
            }
            Operation::AvgPool2d {kernel,stride} => (avg_pool2d_forward_data(x,*kernel,*stride)?,BuiltinBackward::AvgPool2d {kernel:*kernel,stride:*stride}),
            Operation::NearestUpsample2d {scale} => (nearest_upsample2d_forward_data(x,*scale)?,BuiltinBackward::NearestUpsample2d {scale:*scale}),
            Operation::GroupNorm {groups,epsilon} => {
                let (out,s) = group_norm_forward_data(x,&values[1],&values[2],*groups,*epsilon)?;
                saved=s; (out,BuiltinBackward::GroupNorm {groups:*groups,epsilon:*epsilon})
            }
            Operation::Loss {kind,reduction} => {
                if let LossKind::Huber {delta} = kind {
                    if !delta.is_finite() || *delta <= 0.0 { return Err(LossError::InvalidOperation {op:"huber_loss",reason:"delta must be finite and positive".into()}.into()); }
                }
                let (out,s) = loss_forward(*kind,*reduction,inputs[0],inputs[1],true)?;
                saved.extend(s); (out,BuiltinBackward::Loss {kind:*kind,reduction:*reduction})
            }
            Operation::TopK {k,sorted} => {
                let (v,i,s) = topk_forward_data(inputs[0],*k,*sorted)?;
                return Ok(OperationOutput {outputs:vec![TensorBuffer::from_vec(v,&s)?,TensorBuffer::from_vec(i,&s)?],saved,backward:None});
            }
            Operation::Matmax {axis,keepdim} => {
                let (v,i,s) = matmax_forward_data(inputs[0],*axis,*keepdim)?;
                return Ok(OperationOutput {outputs:vec![TensorBuffer::from_vec(v,&s)?,TensorBuffer::from_vec(i,&s)?],saved,backward:None});
            }
        };
        if matches!(operation, Operation::Exp|Operation::Sqrt|Operation::Tanh|Operation::Sigmoid|Operation::Softmax {..}) {
            saved.push(output.clone());
        }
        Ok(OperationOutput { outputs: vec![output], saved, backward: Some(into_node_backward(backward)) })
    }
}
