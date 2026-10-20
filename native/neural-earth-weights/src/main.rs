#[allow(dead_code)]
#[path = "../../magna-urbis/rust/crates/neural/src/weights.rs"]
mod weights;
#[allow(dead_code)]
mod export {
    include!(concat!(env!("OUT_DIR"), "/export.rs"));
}
#[cfg(test)]
#[path = "../../neural-earth/src/packed_base.rs"]
mod packed_base;

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("Usage: neural-earth-weights coarse4.pt decoder.pt output-directory".into());
    }
    export::students(args[0].as_ref(), args[1].as_ref(), args[2].as_ref())
}
