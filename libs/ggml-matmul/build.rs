fn main() {
    let ggml_root = std::env::var("GGML_ROOT").unwrap_or_else(|_| {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let workspace_root = std::path::Path::new(&manifest)
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let sibling = workspace_root.parent().unwrap().join("llama.cpp");
        if sibling.exists() {
            return sibling.to_string_lossy().to_string();
        }
        panic!("llama.cpp not found. Clone it next to the starling workspace.");
    });

    let ggml_lib_dir = format!("{}/build-cpu/ggml/src", ggml_root);
    let ggml_include = format!("{}/ggml/include", ggml_root);
    let ggml_src_include = format!("{}/ggml/src", ggml_root);

    // Compile our C wrapper with ggml headers
    cc::Build::new()
        .file("csrc/ggml_matmul_wrapper.c")
        .include(&ggml_include)
        .include(&ggml_src_include)
        .flag("-O3")
        .flag("-march=native")
        .flag("-fPIC")
        .compile("ggml_matmul_wrapper");

    // Link pre-built ggml static libraries
    println!("cargo:rustc-link-search=native={}", ggml_lib_dir);
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml");
    // ggml uses OpenMP for threading
    println!("cargo:rustc-link-lib=gomp");
    // C++ stdlib (ggml-cpu has C++ components)
    println!("cargo:rustc-link-lib=stdc++");
}
