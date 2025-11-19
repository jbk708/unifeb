fn main() {
    let target = std::env::var("TARGET").unwrap();
    
    // Print target for debugging
    println!("cargo:warning=Building for target: {}", target);
    
    // On macOS (especially ARM64), use macos-accelerate instead of intel-mkl-static
    if target.contains("apple-darwin") {
        println!("cargo:warning=Using macos-accelerate BLAS backend");
    } else if target.contains("x86_64") || target.contains("i686") {
        println!("cargo:warning=Using intel-mkl-static BLAS backend");
    } else {
        println!("cargo:warning=Using default BLAS backend (no BLAS features)");
    }
}
