#[path = "support/legacy_diffusion.rs"]
mod diffusion;
#[path = "support/legacy_operation.rs"]
mod operation;
#[path = "support/legacy_trainer.rs"]
mod trainer;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = serde_json::json!({"operation":operation::run_case(100)?,"unet":diffusion::run_case(30)?,"trainer":trainer::run_case(30)?});
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
