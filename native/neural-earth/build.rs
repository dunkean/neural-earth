// Compile the pinned engine, with the small integration overlay prepared by Node.
// There is only one maintained copy of the network implementations.
fn main() {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("generated-engine");
    let mut modules = String::new();
    // gpu, edm and student are the local direct WebGPU runtime (src/engine).
    for name in ["noise", "pipeline", "plan", "reconstruct", "sketch", "weights"] {
        let path = source.join(format!("{name}.rs"));
        assert!(
            path.exists(),
            "Run npm run build:wasm first (missing {})",
            path.display()
        );
        println!("cargo:rerun-if-changed={}", path.display());
        modules.push_str(&format!(
            "#[path = {:?}] pub mod {name};\n",
            path.to_string_lossy()
        ));
    }
    std::fs::write(
        std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("engine.rs"),
        modules,
    )
    .unwrap();
}
