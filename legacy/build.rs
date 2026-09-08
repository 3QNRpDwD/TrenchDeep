use std::{env, fs, io, path::Path};

fn copy_tree(source: &Path, target: &Path) -> io::Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let destination = target.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &destination)?;
        } else {
            fs::copy(entry.path(), destination)?;
        }
    }
    Ok(())
}
fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=named_parameters");
    let output = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let source = output.join("reference_src");
    copy_tree(Path::new("src"), &source)?;
    for (module, extension) in [
        ("nn/mod.rs", "nn.rs"),
        ("tests/common/model/diffusion/unet.rs", "unet.rs"),
        ("tests/common/model/diffusion/embedding.rs", "embedding.rs"),
    ] {
        use io::Write;
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(source.join(module))?;
        writeln!(
            file,
            "\n{}",
            fs::read_to_string(Path::new("named_parameters").join(extension))?
        )?;
    }
    fs::write(
        output.join("reference_models.rs"),
        format!(
            "#[path = {:?}] pub mod common;",
            source.join("tests/common/mod.rs").to_str().unwrap()
        ),
    )?;
    Ok(())
}
