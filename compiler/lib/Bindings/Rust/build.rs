extern crate bindgen;

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::exit;

const MLIR_STATIC_LIBS: [&str; 179] = [
    "MLIRMemRefDialect",
    "MLIRVectorToSPIRV",
    "MLIRControlFlowInterfaces",
    "MLIRLinalgToStandard",
    "MLIRAnalysis",
    "MLIRSPIRVDeserialization",
    "MLIRTransformDialect",
    "MLIRSparseTensorPipelines",
    "MLIRVectorToGPU",
    "MLIRTranslateLib",
    "MLIRPass",
    "MLIRComplexToLibm",
    "MLIRInferTypeOpInterface",
    "MLIRMemRefToSPIRV",
    "MLIRAMDGPUToROCDL",
    "MLIRBufferizationTransformOps",
    "MLIRExecutionEngineUtils",
    "MLIRNVVMDialect",
    "MLIRSCFUtils",
    "MLIRLinalgTransforms",
    "MLIRParser",
    "MLIRFuncTransforms",
    "MLIRTosaTestPasses",
    "MLIRTosaToArith",
    "MLIRTensorDialect",
    "MLIRGPUTransforms",
    "MLIRLowerableDialectsToLLVM",
    "MLIRBufferizationToMemRef",
    "MLIRPresburger",
    "MLIRFuncDialect",
    "MLIRPDLToPDLInterp",
    "MLIRArithmeticTransforms",
    "MLIRViewLikeInterface",
    "MLIRTargetCpp",
    "MLIROpenMPToLLVM",
    "MLIRSPIRVConversion",
    "MLIRNVGPUTransforms",
    "MLIRSparseTensorTransforms",
    "MLIRAffineAnalysis",
    "MLIRArmSVETransforms",
    "MLIRArmNeon2dToIntr",
    "MLIRDataLayoutInterfaces",
    "MLIRAffineTransforms",
    "MLIROpenACCToLLVMIRTranslation",
    "MLIRTensorUtils",
    "MLIRSPIRVSerialization",
    "MLIRShapeToStandard",
    "MLIRArithmeticToSPIRV",
    "MLIRArithmeticDialect",
    "MLIRFuncToSPIRV",
    "MLIRQuantUtils",
    "MLIRTensorTilingInterfaceImpl",
    "MLIRX86VectorToLLVMIRTranslation",
    "MLIRCopyOpInterface",
    "MLIRMathToLibm",
    "MLIRGPUToGPURuntimeTransforms",
    "MLIRLLVMDialect",
    "MLIRAffineDialect",
    "MLIRTransforms",
    "MLIRVectorTransforms",
    "MLIROpenMPDialect",
    "MLIRControlFlowDialect",
    "MLIRVectorUtils",
    "MLIRROCDLDialect",
    "MLIRPDLDialect",
    "MLIRAsyncDialect",
    "MLIRLinalgToLLVM",
    "MLIROpenACCDialect",
    "MLIRVectorDialect",
    "MLIROpenACCToSCF",
    "MLIRIR",
    "MLIRCAPIIR",
    "MLIRTargetLLVMIRImport",
    "MLIRTensorToLinalg",
    "MLIRCallInterfaces",
    "MLIRTensorInferTypeOpInterfaceImpl",
    "MLIRTransformDialectTransforms",
    "MLIRComplexDialect",
    "MLIRAffineUtils",
    "MLIRLoopLikeInterface",
    "MLIRDialect",
    "MLIRLinalgUtils",
    "MLIRSCFToSPIRV",
    "MLIRAffineToStandard",
    "MLIRX86VectorDialect",
    "MLIRGPUToVulkanTransforms",
    "MLIRRewrite",
    "MLIRAMXToLLVMIRTranslation",
    "MLIRInferIntRangeInterface",
    "MLIRCAPIRegistration",
    "MLIRNVVMToLLVMIRTranslation",
    "MLIRAsyncTransforms",
    "MLIRPDLInterpDialect",
    "MLIRTransformUtils",
    "MLIRLinalgDialect",
    "MLIRMathDialect",
    "MLIRMemRefTransforms",
    "MLIRSPIRVModuleCombiner",
    "MLIRMathToLLVM",
    "MLIRControlFlowToLLVM",
    "MLIRArmSVEDialect",
    "MLIRSPIRVTranslateRegistration",
    "MLIRToLLVMIRTranslationRegistration",
    "MLIRSCFDialect",
    "MLIRTilingInterface",
    "MLIREmitCDialect",
    "MLIRTableGen",
    "MLIRTosaToSCF",
    "MLIROpenMPToLLVMIRTranslation",
    "MLIRSupport",
    "MLIROpenACCToLLVM",
    "MLIRAMDGPUDialect",
    "MLIRTosaToLinalg",
    "MLIRSparseTensorUtils",
    "MLIRFuncToLLVM",
    "MLIRTargetLLVMIRExport",
    "MLIRControlFlowToSPIRV",
    "MLIRReconcileUnrealizedCasts",
    "MLIRComplexToStandard",
    "MLIRMathTransforms",
    "MLIRSPIRVUtils",
    "MLIRCastInterfaces",
    "MLIRTosaToTensor",
    "MLIRMemRefUtils",
    "MLIRGPUToSPIRV",
    "MLIRBufferizationDialect",
    "MLIRSCFToControlFlow",
    "MLIRArmSVEToLLVMIRTranslation",
    "MLIRExecutionEngine",
    "MLIRBufferizationTransforms",
    "MLIRSparseTensorDialect",
    "MLIRTensorToSPIRV",
    "MLIRVectorToSCF",
    "MLIRQuantTransforms",
    "MLIRLLVMToLLVMIRTranslation",
    "MLIRNVGPUDialect",
    "MLIRAsyncToLLVM",
    "MLIRAMXDialect",
    "MLIRLinalgTransformOps",
    "MLIRMathToSPIRV",
    "MLIRSCFToOpenMP",
    "MLIRShapeDialect",
    "MLIRGPUToROCDLTransforms",
    "MLIRGPUToNVVMTransforms",
    "MLIRTensorTransforms",
    "MLIRSCFToGPU",
    "MLIRDialectUtils",
    "MLIRNVGPUToNVVM",
    "MLIRTosaDialect",
    "MLIRVectorToLLVM",
    "MLIRSPIRVDialect",
    "MLIRSideEffectInterfaces",
    "MLIRVectorToROCDL",
    "MLIRQuantDialect",
    "MLIRSCFTransforms",
    "MLIRMLProgramDialect",
    "MLIRLinalgToSPIRV",
    "MLIRDLTIDialect",
    "MLIRLinalgFrontend",
    "MLIRROCDLToLLVMIRTranslation",
    "MLIRArmNeonDialect",
    "MLIRSPIRVToLLVM",
    "MLIRLLVMIRTransforms",
    "MLIRTosaTransforms",
    "MLIRLLVMCommonConversion",
    "MLIRSCFTransformOps",
    "MLIRArmNeonToLLVMIRTranslation",
    "MLIRAMXTransforms",
    "MLIRSPIRVTransforms",
    "MLIRMemRefToLLVM",
    "MLIRSPIRVBinaryUtils",
    "MLIRLinalgAnalysis",
    "MLIRArithmeticUtils",
    "MLIRVectorInterfaces",
    "MLIRGPUOps",
    "MLIRComplexToLLVM",
    "MLIRShapeOpsTransforms",
    "MLIRX86VectorTransforms",
    "MLIRArithmeticToLLVM",
];

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    // library paths
    let build_dir = get_build_dir();
    let lib_dir = build_dir.join("lib");
    println!("cargo:rustc-link-search={}", lib_dir.to_str().unwrap());

    // include paths
    let root = std::fs::canonicalize("../../../../")?;
    let include_paths = [
        // compiler build
        build_dir.join("tools/concretelang/include/"),
        // mlir build
        build_dir.join("tools/mlir/include"),
        // llvm build
        build_dir.join("include"),
        // compiler
        root.join("compiler/include/"),
        // mlir
        root.join("llvm-project/mlir/include/"),
        // llvm
        root.join("llvm-project/llvm/include/"),
        // concrete-optimizer
        root.join("compiler/concrete-optimizer/concrete-optimizer-cpp/src/cpp/"),
    ];

    // linking
    for mlir_static_lib in MLIR_STATIC_LIBS {
        println!("cargo:rustc-link-lib=static={}", mlir_static_lib);
    }
    println!("cargo:rustc-link-lib=static=LLVMSupport");
    println!("cargo:rustc-link-lib=static=LLVMCore");
    // required by llvm
    println!("cargo:rustc-link-lib=tinfo");
    if let Some(name) = get_system_libcpp() {
        println!("cargo:rustc-link-lib={}", name);
    }
    // concrete-compiler libs
    println!("cargo:rustc-link-lib=static=CONCRETELANGCAPIFHE");
    println!("cargo:rustc-link-lib=static=FHEDialect");

    println!("cargo:rerun-if-changed=api.h");
    bindgen::builder()
        .header("api.h")
        .clang_args(
            include_paths
                .into_iter()
                .map(|path| format!("-I{}", path.to_str().unwrap())),
        )
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

fn get_system_libcpp() -> Option<&'static str> {
    if cfg!(target_env = "msvc") {
        None
    } else if cfg!(target_os = "macos") {
        Some("c++")
    } else {
        Some("stdc++")
    }
}

fn get_build_dir() -> PathBuf {
    // this env variable can be used to point to a different build directory
    let build_dir = match env::var("CONCRETE_COMPILER_BUILD_DIR") {
        Ok(val) => std::path::Path::new(&val).to_path_buf(),
        Err(_e) => std::path::Path::new(".")
            .parent()
            .unwrap()
            .join("..")
            .join("..")
            .join("..")
            .join("build"),
    };
    return build_dir;
}
