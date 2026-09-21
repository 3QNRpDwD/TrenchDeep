#![cfg(all(feature="builtinStorage", feature="builtinKernels", feature="enableBackward"))]
mod support;
use trench_deep::{*, optimizer::*, contracts::Operation, runtime::prepared::*};
fn optimizer(ctx:&ExecutionContext, kind:usize)->MlResult<Box<dyn Optimizer>> {
    Ok(match kind {
        0=>Box::new(SGD::new(ctx,0.01)?),
        1=>Box::new(Momentum::new(ctx,0.01,0.9)?),
        2=>Box::new(AdaGrad::new(ctx,0.01,1e-8)?),
        3=>Box::new(RMSProp::new(ctx,0.01,0.9,1e-8)?),
        4=>Box::new(Adam::new(ctx,0.01,0.9,0.999,1e-8)?),
        _=>Box::new(AdamW::new(ctx,0.01,0.9,0.999,1e-8,0.1)?),
    })
}
#[test]
fn direct_and_external_store_updates_match_and_rejected_steps_do_not_advance_adam() -> MlResult<()> {
    for kind in 0..6 {
        let direct=ExecutionContext::new();
        let fallback=ExecutionContext::builder().storage(support::SlotStore::default()).build();
        let a=direct.parameter(vec![0.5,-1.,2.],&[3])?;
        let b=fallback.parameter(vec![0.5,-1.,2.],&[3])?;
        let mut oa=optimizer(&direct,kind)?; oa.register(&a)?;
        let mut ob=optimizer(&fallback,kind)?; ob.register(&b)?;
        let mut p=PreparedProgram::new();
        let w=p.parameter(&[3])?;
        let square=p.operation(Operation::Square,&[w])?;
        let loss=p.operation(Operation::Sum,&[square])?;
        let mut ea=direct.prepare(&p,&[&a],&[loss],PreparedMode::Training)?.into_executor(&direct)?;
        let mut eb=fallback.prepare(&p,&[&b],&[loss],PreparedMode::Training)?.into_executor(&fallback)?;
        for _ in 0..20 {
            ea.with_run(&[],&[&a],|out| {
                assert!(oa.step().is_err());
                out[0].as_variable()?.backward()?;
                oa.step()
            })?;
            eb.with_run(&[],&[&b],|out| {out[0].as_variable()?.backward()?;ob.step()})?;
            assert_eq!(a.tensor().to_vec()?,b.tensor().to_vec()?);
        }
    }
    Ok(())
}
