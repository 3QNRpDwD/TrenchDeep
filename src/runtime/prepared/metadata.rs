//! Owned untyped storage for call-scoped, non-owning metadata. No arena
//! references survive in a typed container between calls.
use super::invalid;
use crate::{MlResult, TensorView, contracts::GradientWrite};
use std::mem::{MaybeUninit, align_of, needs_drop, size_of};

#[derive(Debug)]
pub(super) struct Scratch(Vec<MaybeUninit<usize>>);
impl Scratch {
    pub fn new<T>(count: usize) -> MlResult<Self> {
        let bytes = count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| invalid("metadata overflow"))?;
        let words = bytes.div_ceil(size_of::<usize>());
        let mut data = Vec::new();
        data.try_reserve_exact(words)
            .map_err(|_| invalid("metadata allocation failed"))?;
        data.resize_with(words, MaybeUninit::uninit);
        Ok(Self(data))
    }
    pub fn with<T, R>(
        &mut self,
        count: usize,
        mut init: impl FnMut(usize) -> MlResult<T>,
        operation: impl FnOnce(&mut [T]) -> MlResult<R>,
    ) -> MlResult<R> {
        if needs_drop::<T>()
            || size_of::<T>() == 0
            || align_of::<T>() > align_of::<usize>()
            || count
                .checked_mul(size_of::<T>())
                .is_none_or(|n| n > self.0.len() * size_of::<usize>())
        {
            return Err(invalid("invalid metadata scratch layout"));
        }
        let ptr = self.0.as_mut_ptr().cast::<T>();
        // SAFETY: checked size/alignment; every element is initialized before
        // exposing the slice. T has no destructor, so init errors and panics
        // leave only inert bytes. The callback cannot retain a reference to
        // this slice; any arena references inside T keep their original lifetime.
        // Exclusive self prevents reuse while the callback runs. No typed T is
        // read or dropped once the callback exits, even during unwinding.
        unsafe {
            for i in 0..count {
                ptr.add(i).write(init(i)?);
            }
            operation(std::slice::from_raw_parts_mut(ptr, count))
        }
    }
}

#[derive(Debug)]
pub(super) struct Metadata {
    pub io_inputs: Scratch,
    pub io_outputs: Scratch,
    pub inputs: Scratch,
    pub saved: Scratch,
    pub outputs: Scratch,
    pub writes: Scratch,
}
impl Metadata {
    pub fn new(inputs: usize, saved: usize) -> MlResult<Self> {
        let reads = inputs
            .checked_add(saved)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("metadata overflow"))?;
        let writes = inputs.max(
            saved
                .checked_add(1)
                .ok_or_else(|| invalid("metadata overflow"))?,
        );
        Ok(Self {
            io_inputs: Scratch::new::<&[f32]>(reads)?,
            io_outputs: Scratch::new::<&mut [f32]>(writes)?,
            inputs: Scratch::new::<Option<TensorView<'_>>>(inputs)?,
            saved: Scratch::new::<TensorView<'_>>(saved)?,
            outputs: Scratch::new::<Option<&mut [f32]>>(inputs.max(saved))?,
            writes: Scratch::new::<GradientWrite>(inputs)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_reuses_after_init_error_and_callback_panic() -> MlResult<()> {
        let mut scratch = Scratch::new::<&mut [f32]>(4)?;
        let mut values = [1., 2., 3., 4.];
        let result = scratch.with(
            4,
            |i| if i == 2 { Err(invalid("init")) } else { Ok(i) },
            |_| Ok(()),
        );
        assert!(result.is_err());
        let mut chunks = values.chunks_mut(1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scratch.with(
                4,
                |_| Ok(chunks.next().unwrap()),
                |slices| -> MlResult<()> {
                    slices[0][0] = 7.;
                    panic!("callback");
                },
            )
        }));
        assert!(result.is_err());
        drop(chunks);
        let mut chunks = values.chunks_mut(1);
        scratch.with(
            4,
            |_| Ok(chunks.next().unwrap()),
            |slices| {
                for slice in slices {
                    slice[0] += 1.;
                }
                Ok(())
            },
        )?;
        assert_eq!(values, [8., 3., 4., 5.]);
        assert!(scratch.with(5, |_| Ok(&[] as &[f32]), |_| Ok(())).is_err());
        Ok(())
    }
}
