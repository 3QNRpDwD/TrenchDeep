use super::*;

pub mod mlp;

pub struct MLP {
    pub layer: Sequential,
    loss_function: GlobalFunction,
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP {{")?;
        writeln!(f, "  layer = {:?}", self.layer)?;
        writeln!(f, "  loss_function = {}", self.loss_function.name())?;
        writeln!(f, "}}")
    }
}
