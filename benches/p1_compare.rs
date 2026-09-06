#[path = "../tests/legacy_diffusion.rs"]
mod diffusion;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = diffusion::run_case(30)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
