#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use std::{cell::Cell, rc::Rc};
use trench_deep::{contracts::*, runtime::prepared::*, *};

#[derive(Debug)]
struct Provider {
    calls: Rc<Cell<usize>>,
    alias: Option<usize>,
}
#[derive(Debug)]
struct Kernel {
    inner: Rc<dyn IntoKernel>,
    calls: Rc<Cell<usize>>,
    alias: Option<usize>,
}
impl IntoKernel for Kernel {
    fn spec(&self) -> &IntoKernelSpec {
        self.inner.spec()
    }
    fn forward_alias(&self) -> Option<usize> {
        self.alias
    }
    fn execute_into(
        &self,
        x: &[TensorView<'_>],
        y: &mut [f32],
        s: &mut [&mut [f32]],
        w: &mut [f32],
    ) -> MlResult<()> {
        self.calls.set(self.calls.get() + 1);
        self.inner.execute_into(x, y, s, w)
    }
    fn backward_into(
        &self,
        i: usize,
        x: &[Option<TensorView<'_>>],
        s: &[TensorView<'_>],
        g: TensorView<'_>,
        d: &mut [f32],
        mode: GradientWrite,
        w: &mut [f32],
    ) -> MlResult<()> {
        self.inner.backward_into(i, x, s, g, d, mode, w)
    }
}
impl OperationProvider for Provider {
    fn supports_prepared_replay(&self) -> bool {
        true
    }
    fn execute(&self, op: &Operation, x: &[TensorView<'_>]) -> MlResult<OperationOutput> {
        backend::CpuBackend::default().execute(op, x)
    }
    fn prepare_backward(
        &self,
        op: &Operation,
        s: &[&[usize]],
    ) -> MlResult<Option<PreparedBackward>> {
        backend::CpuBackend::default().prepare_backward(op, s)
    }
    fn prepare_into(
        &self,
        op: &Operation,
        s: &[&[usize]],
        t: bool,
    ) -> MlResult<Option<Rc<dyn IntoKernel>>> {
        Ok(backend::CpuBackend::default()
            .prepare_into(op, s, t)?
            .map(|inner| {
                Rc::new(Kernel {
                    inner,
                    calls: self.calls.clone(),
                    alias: self.alias,
                }) as Rc<dyn IntoKernel>
            }))
    }
}
#[test]
fn provider_must_explicitly_allow_skipping_forward() -> MlResult<()> {
    for alias in [None, Some(0), Some(1)] {
        let calls = Rc::new(Cell::new(0));
        let ctx = ExecutionContext::builder()
            .operations(Provider {
                calls: calls.clone(),
                alias,
            })
            .build();
        let mut p = PreparedProgram::new();
        let x = p.input(&[2, 3], false)?;
        let y = p.operation(Operation::Reshape(vec![6]), &[x])?;
        let result = ctx
            .prepare(&p, &[], &[y], PreparedMode::Inference)?
            .into_executor(&ctx);
        if alias == Some(1) {
            assert!(result.is_err());
            continue;
        }
        let mut e = result?;
        let x = ctx.tensor(vec![2.; 6], &[2, 3])?;
        e.with_run(&[&x], &[], |out| {
            assert_eq!(out[0].to_vec()?, vec![2.; 6]);
            Ok(())
        })?;
        assert_eq!(calls.get(), if alias.is_none() { 1 } else { 0 });
    }
    Ok(())
}
