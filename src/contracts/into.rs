//! Optional, allocation-free numerical kernel contract. Metadata is owned at
//! preparation time; borrowed views and destinations never escape a call.
use super::*;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntoKernelSpec {
    pub inputs: Vec<Vec<usize>>,
    pub output: Vec<usize>,
    pub saved: Vec<Vec<usize>>,
    pub backward_inputs: Vec<BackwardInput>,
    pub workspace_elements: usize,
    pub training: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientWrite {
    Assign,
    Add,
}
pub trait IntoKernel: Debug {
    fn spec(&self) -> &IntoKernelSpec;
    /// Opt in to a metadata-only forward: output has exactly the selected
    /// input's contiguous elements, with no saved values or forward side effects.
    /// The executor may skip execute_into. Gradients remain independent.
    fn forward_alias(&self) -> Option<usize> {
        None
    }
    /// Fully overwrite output and saved destinations. Validate all lengths
    /// before writing. Successful numerical calls allocate no tensor data.
    fn execute_into(
        &self,
        inputs: &[TensorView<'_>],
        output: &mut [f32],
        saved: &mut [&mut [f32]],
        workspace: &mut [f32],
    ) -> MlResult<()>;
    /// Compute one input contribution. Separate destinations allow shared
    /// parameters to accumulate without simultaneous mutable aliases.
    /// None inputs are allowed only for Shape dependencies. Assign replaces
    /// the complete destination; Add adds the already-reduced contribution.
    fn backward_into(
        &self,
        input: usize,
        inputs: &[Option<TensorView<'_>>],
        saved: &[TensorView<'_>],
        gradient: TensorView<'_>,
        destination: &mut [f32],
        write: GradientWrite,
        workspace: &mut [f32],
    ) -> MlResult<()>;

    /// Compute multiple input contributions into disjoint destinations. Entries
    /// follow input order; None skips an untracked input. Callers must split
    /// aliased destinations into separate calls, preserving contribution order.
    /// Providers may fuse reductions; the default preserves the single-input ABI.
    fn backward_many_into(
        &self,
        inputs: &[Option<TensorView<'_>>],
        saved: &[TensorView<'_>],
        gradient: TensorView<'_>,
        destinations: &mut [Option<&mut [f32]>],
        writes: &[GradientWrite],
        workspace: &mut [f32],
    ) -> MlResult<()> {
        if destinations.len() != self.spec().inputs.len()
            || writes.len() != destinations.len()
            || destinations.iter().zip(&self.spec().inputs).any(|(d, s)| {
                d.as_ref()
                    .is_some_and(|d| d.len() != s.iter().product::<usize>())
            })
        {
            return Err(crate::TensorError::InvalidOperation {
                op: "backward_many_into",
                reason: "gradient destination arity or shape mismatch".into(),
            }
            .into());
        }
        for (input, destination) in destinations.iter_mut().enumerate() {
            if let Some(destination) = destination {
                self.backward_into(
                    input,
                    inputs,
                    saved,
                    gradient,
                    destination,
                    writes[input],
                    workspace,
                )?;
            }
        }
        Ok(())
    }
}
