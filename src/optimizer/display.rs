use super::*;

impl Display for OptimError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimError::GradientError(msg) => write!(f, "Gradient error: {}", msg),
        }
    }
}