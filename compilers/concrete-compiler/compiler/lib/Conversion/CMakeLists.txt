add_compile_options(-fexceptions)

add_subdirectory(FHEToTFHEScalar)
add_subdirectory(FHEToTFHECrt)
add_subdirectory(TFHEGlobalParametrization)
add_subdirectory(TFHEKeyNormalization)
add_subdirectory(SimulateTFHE)
add_subdirectory(TFHEToConcrete)
add_subdirectory(FHETensorOpsToLinalg)
add_subdirectory(TracingToCAPI)
add_subdirectory(ConcreteToCAPI)
add_subdirectory(SDFGToStreamEmulator)
add_subdirectory(MLIRLowerableDialectsToLLVM)
add_subdirectory(LinalgExtras)
add_subdirectory(ExtractSDFGOps)
add_subdirectory(Utils)

add_mlir_library(
  ConcretelangConversion
  Tools.cpp
  Utils/Dialects/SCF.cpp
  Utils/Dialects/Tensor.cpp
  Utils/Utils.cpp
  LINK_LIBS
  PUBLIC
  MLIRIR)
