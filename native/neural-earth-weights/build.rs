fn main() {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../magna-urbis/rust/crates/neural/src/export.rs");
    println!("cargo:rerun-if-changed={}", source.display());
    let mut code = std::fs::read_to_string(source)
        .unwrap()
        .replace("//!", "//");
    // Reuse the native exporter's readers and model preparation, exposing only
    // the two requested student checkpoints. No original base is ever read.
    code.push_str(r#"
pub fn students(coarse: &Path, decoder: &Path, out: &Path) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let coarse = prepare_edm(read_torch(coarse, "model")?, "solver.net.", "coarse4");
    std::fs::write(out.join("coarse.safetensors"), coarse.serialize()).map_err(|e| e.to_string())?;
    let decoder = prepare_student(read_torch(decoder, "model")?, "student200k");
    std::fs::write(out.join("decoder.safetensors"), decoder.serialize()).map_err(|e| e.to_string())?;
    Ok(())
}
"#);
    std::fs::write(
        std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("export.rs"),
        code,
    )
    .unwrap();
}
