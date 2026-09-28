fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(target_arch = "wasm32")]
    {
        if !cfg!(target_feature = "simd128") {
            println!("cargo:warning=[sys1pop-core] SIMD128 target feature is not enabled for wasm32!");
            println!("cargo:warning=[sys1pop-core] To achieve full inference performance, add the following to your .cargo/config.toml:");
            println!("cargo:warning=[sys1pop-core] [target.wasm32-unknown-unknown]");
            println!("cargo:warning=[sys1pop-core] rustflags = [\"-C\", \"target-feature=+simd128\"]");
        }
    }
}
