extern crate bindgen;

use std::env;
use std::error::Error;
use std::path::Path;
use std::process::exit;

const MLIR_STATIC_LIBS: &[&str] = &[
    "MLIRArithAttrToLLVMConversion",
    "MLIRDestinationStyleOpInterface",
    "MLIRVectorTransformOps",
    "MLIRMemRefTransformOps",
    "MLIRGPUTransformOps",
    "MLIRAffineTransformOps",
    "MLIRBytecodeReader",
    "MLIRAsmParser",
    "MLIRIndexDialect",
    "MLIRMaskableOpInterface",
    "MLIRMaskingOpInterface",
    "MLIRInferIntRangeCommon",
    "MLIRShapedOpInterfaces",
    "MLIRTransformDialectUtils",
    "MLIRParallelCombiningOpInterface",
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
    "MLIRArithTransforms",
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
    "MLIRArithToSPIRV",
    "MLIRArithDialect",
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
    "MLIRCAPIRegisterEverything",
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
    "MLIRGPUToSPIRV",
    "MLIRBufferizationDialect",
    "MLIRSCFToControlFlow",
    "MLIRArmSVEToLLVMIRTranslation",
    "MLIRExecutionEngine",
    "MLIRBufferizationTransforms",
    "MLIRSparseTensorDialect",
    "MLIRTensorToSPIRV",
    "MLIRVectorToSCF",
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
    "MLIRQuantDialect",
    "MLIRSCFTransforms",
    "MLIRMLProgramDialect",
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
    "MLIRArithUtils",
    "MLIRVectorInterfaces",
    "MLIRGPUOps",
    "MLIRComplexToLLVM",
    "MLIRShapeOpsTransforms",
    "MLIRX86VectorTransforms",
    "MLIRArithToLLVM",
];

const LLVM_STATIC_LIBS: &[&str] = &[
    "LLVMAggressiveInstCombine",
    "LLVMAnalysis",
    "LLVMAsmParser",
    "LLVMAsmPrinter",
    "LLVMBinaryFormat",
    "LLVMBitReader",
    "LLVMBitstreamReader",
    "LLVMBitWriter",
    "LLVMCFGuard",
    "LLVMCodeGen",
    "LLVMCore",
    "LLVMCoroutines",
    "LLVMDebugInfoCodeView",
    "LLVMDebugInfoDWARF",
    "LLVMDebugInfoMSF",
    "LLVMDebugInfoPDB",
    "LLVMDemangle",
    "LLVMExecutionEngine",
    "LLVMFrontendOpenMP",
    "LLVMGlobalISel",
    "LLVMInstCombine",
    "LLVMInstrumentation",
    "LLVMipo",
    "LLVMIRReader",
    "LLVMJITLink",
    "LLVMLinker",
    "LLVMMC",
    "LLVMMCDisassembler",
    "LLVMMCParser",
    "LLVMObjCARCOpts",
    "LLVMObject",
    "LLVMOption",
    "LLVMOrcJIT",
    "LLVMOrcShared",
    "LLVMOrcTargetProcess",
    "LLVMPasses",
    "LLVMProfileData",
    "LLVMRemarks",
    "LLVMRuntimeDyld",
    "LLVMScalarOpts",
    "LLVMSelectionDAG",
    "LLVMSupport",
    "LLVMSymbolize",
    "LLVMTableGen",
    "LLVMTableGenGlobalISel",
    "LLVMTarget",
    "LLVMTargetParser",
    "LLVMTextAPI",
    "LLVMTransformUtils",
    "LLVMVectorize",
];

#[cfg(target_arch = "aarch64")]
const LLVM_TARGET_SPECIFIC_STATIC_LIBS: &[&str] = &[
    "LLVMAArch64Utils",
    "LLVMAArch64Info",
    "LLVMAArch64Desc",
    "LLVMAArch64CodeGen",
];

#[cfg(target_arch = "x86_64")]
const LLVM_TARGET_SPECIFIC_STATIC_LIBS: &[&str] = &["LLVMX86CodeGen", "LLVMX86Desc", "LLVMX86Info"];

const CONCRETE_COMPILER_STATIC_LIBS: &[&str] = &[
    "AnalysisUtils",
    "RTDialect",
    "RTDialectTransforms",
    "ConcretelangSupport",
    "ConcreteToCAPI",
    "ConcretelangConversion",
    "ConcretelangTransforms",
    "FHETensorOpsToLinalg",
    "ConcretelangServerLib",
    "CONCRETELANGCAPIFHE",
    "TFHEGlobalParametrization",
    "ConcretelangClientLib",
    "ConcretelangConcreteTransforms",
    "ConcretelangSDFGInterfaces",
    "ConcretelangSDFGTransforms",
    "CONCRETELANGCAPISupport",
    "FHELinalgDialect",
    "TracingDialect",
    "TracingDialectTransforms",
    "TracingToCAPI",
    "ConcretelangInterfaces",
    "TFHEDialect",
    "SimulateTFHE",
    "CONCRETELANGCAPIFHELINALG",
    "FHELinalgDialectTransforms",
    "FHEDialect",
    "FHEDialectTransforms",
    "TFHEToConcrete",
    "FHEToTFHECrt",
    "FHEToTFHEScalar",
    "TFHEDialectTransforms",
    "TFHEKeyNormalization",
    "concrete_optimizer",
    "LinalgExtras",
    "FHEDialectAnalysis",
    "ConcreteDialect",
    "RTDialectAnalysis",
    "SDFGDialect",
    "ExtractSDFGOps",
    "SDFGToStreamEmulator",
    "TFHEDialectAnalysis",
    "ConcreteDialectAnalysis",
];

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut include_paths = Vec::new();
    // if set, use installation path of concrete compiler to lookup libraries and include files
    match env::var("CONCRETE_COMPILER_INSTALL_DIR") {
        Ok(install_dir) => {
            println!("cargo:rustc-link-search={}/lib/", install_dir);
            include_paths.push(Path::new(&format!("{}/include/", install_dir)).to_path_buf());
        }
        Err(_e) => println!(
            "cargo:warning=You are not setting CONCRETE_COMPILER_INSTALL_DIR, \
so your compiler/linker will have to lookup libs and include dirs on their own"
        ),
    }
    // linking to static libs
    let all_static_libs = CONCRETE_COMPILER_STATIC_LIBS
        .into_iter()
        .chain(MLIR_STATIC_LIBS)
        .chain(LLVM_STATIC_LIBS)
        .chain(LLVM_TARGET_SPECIFIC_STATIC_LIBS);

    for static_lib_name in all_static_libs {
        println!("cargo:rustc-link-lib=static={}", static_lib_name);
    }
    // concrete compiler runtime
    println!("cargo:rustc-link-lib=ConcretelangRuntime");
    // concrete optimizer
    // `-bundle` serve to not have multiple definition issues
    println!("cargo:rustc-link-lib=static:-bundle=concrete_optimizer_cpp");

    // required by llvm
    println!("cargo:rustc-link-lib=ncurses");
    if let Some(name) = get_system_libcpp() {
        println!("cargo:rustc-link-lib={}", name);
    }
    // zlib
    println!("cargo:rustc-link-lib=z");

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
