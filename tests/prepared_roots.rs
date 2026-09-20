#![cfg(all(
    feature = "builtinStorage",
    feature = "builtinKernels",
    feature = "enableBackward"
))]
use trench_deep::{contracts::Operation, runtime::prepared::*, *};

#[test]
fn selected_root_preserves_prediction_retention_and_rejected_backward_is_retryable() -> MlResult<()>
{
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2.0, 3.0], &[2])?;
    let mut program = PreparedProgram::new();
    let p = program.parameter(&[2])?;
    let prediction = program.operation(Operation::Square, &[p])?;
    let loss = program.operation(Operation::Sum, &[prediction])?;
    for into in [false, true] {
        let plan = ctx.prepare_with_roots(
            &program,
            &[&w],
            &[loss, prediction],
            &[loss],
            PreparedMode::Training,
        )?;
        assert_eq!(plan.backward_plan_stats().roots, 1);
        let verify = |out: &[Tensor]| -> MlResult<()> {
            assert_eq!(out[1].to_vec()?, vec![4.0, 9.0]);
            let pred = out[1].as_variable()?;
            pred.retain_grad()?;
            let seed = ctx.tensor(vec![2.5, 2.5], &[2])?;
            assert!(pred.backward_with_grad(&seed).is_err());
            assert!(w.grad()?.is_none());
            out[0].as_variable()?.backward()?;
            assert_eq!(w.grad()?.unwrap().data(), &[4.0, 6.0]);
            assert_eq!(pred.grad()?.unwrap().data(), &[1.0, 1.0]);
            Ok(())
        };
        if into {
            let mut executor = plan.into_executor(&ctx)?;
            executor.with_run(&[], &[&w], verify)?;
            executor.with_run(&[], &[&w], verify)?;
        } else {
            plan.with_run(&ctx, &[], &[&w], verify)?;
        }
        assert!(w.grad()?.is_none());
    }
    // Legacy selection behavior and duplicate exports remain supported.
    let plan = ctx.prepare(
        &program,
        &[&w],
        &[loss, prediction, prediction],
        PreparedMode::Training,
    )?;
    assert_eq!(plan.backward_plan_stats().roots, 2);
    let mut executor = plan.into_executor(&ctx)?;
    executor.with_run(&[], &[&w], |out| {
        assert_eq!(out[1].id(), out[2].id());
        let seed = ctx.tensor(vec![2.5, 2.5], &[2])?;
        out[1].as_variable()?.backward_with_grad(&seed)?;
        assert_eq!(w.grad()?.unwrap().data(), &[10.0, 15.0]);
        Ok(())
    })?;
    assert!(
        ctx.prepare_with_roots(
            &program,
            &[&w],
            &[loss],
            &[prediction],
            PreparedMode::Training
        )
        .is_err()
    );
    assert!(
        ctx.prepare_with_roots(
            &program,
            &[&w],
            &[loss],
            &[loss, loss],
            PreparedMode::Training
        )
        .is_err()
    );
    assert!(
        ctx.prepare_with_roots(&program, &[&w], &[loss], &[loss], PreparedMode::Inference)
            .is_err()
    );
    Ok(())
}

#[test]
fn named_root_indices_validate_and_keep_all_exports() -> MlResult<()> {
    let ctx = ExecutionContext::new();
    let w = ctx.parameter(vec![2.0], &[])?;
    let inputs = ExecutionInputs::new("roots");
    let describe = |_: &ExecutionInputs| Ok(vec![w.tensor().square()?]);
    for indices in [&[1][..], &[0, 0][..]] {
        assert!(
            ctx.prepare_forward_with_roots(
                &inputs,
                &[&w],
                PreparedMode::Training,
                indices,
                describe
            )
            .is_err()
        );
    }
    let mut executor = ctx
        .prepare_forward_with_roots(&inputs, &[&w], PreparedMode::Training, &[0], describe)?
        .into_executor(&ctx)?;
    executor.with_inputs(&inputs, &[&w], |out| {
        out[0].as_variable()?.backward()?;
        assert_eq!(w.grad()?.unwrap().data(), &[4.0]);
        Ok(())
    })?;
    Ok(())
}

#[test]
fn sorted_lifetime_validation_matches_pairwise_reference_and_rechecks_public_mutations()
-> MlResult<()> {
    let make = |intervals: &[(usize, usize, usize)]| {
        let lifetimes = intervals
            .iter()
            .enumerate()
            .map(|(i, &(buffer, first, last))| BufferLifetime {
                value: BufferValue::Tensor(i),
                role: BufferRole::Temporary,
                shape: vec![1],
                first,
                last,
                buffer,
            })
            .collect();
        BufferPlan {
            aliases: vec![],
            roots: vec![],
            capacity_elements: vec![1; 3],
            capacity_bytes: 12,
            layout: RootBufferPlan {
                root: None,
                lifetimes,
                buffers: vec![BufferSlot { elements: 1 }; 3],
                copies: vec![],
                arena_bytes: 12,
                unreused_bytes: intervals.len() * 4,
            },
        }
    };
    // Includes unsorted intervals, containment, equal/inclusive endpoints and
    // independent buffers. Reference is intentionally the old quadratic rule.
    for seed in 0..100usize {
        let intervals = (0..8)
            .map(|i| {
                let first = (seed * 7 + i * 11) % 31;
                ((seed + i) % 3, first, first + (seed + i * 3) % 7)
            })
            .collect::<Vec<_>>();
        let overlaps = intervals.iter().enumerate().any(|(i, a)| {
            intervals[..i]
                .iter()
                .any(|b| a.0 == b.0 && a.1 <= b.2 && b.1 <= a.2)
        });
        assert_eq!(make(&intervals).validate().is_err(), overlaps);
    }
    let mut valid = make(&[(0, 0, 2), (0, 3, 4), (1, 0, 4)]);
    valid.allocate_arena()?;
    valid.layout.lifetimes[1].first = 2;
    assert!(valid.allocate_arena().is_err());
    let mut invalid_capacity = make(&[(0, 0, 1)]);
    invalid_capacity.capacity_elements[0] = 0;
    assert!(invalid_capacity.allocate_arena().is_err());
    Ok(())
}
