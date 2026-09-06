use super::*;

impl std::error::Error for LossError {}

impl Display for LossError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LossError::InvalidShape { expected, got } => {
                write!(f, "Invalid shape: expected {:?}, got {:?}", expected, got)
            }
            LossError::InvalidOperation { op, reason } => {
                write!(f, "Invalid operation: {} ({})", op, reason)
            }
        }
    }
}