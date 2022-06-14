add_subdirectory(FHEToTFHE)
add_subdirectory(TFHEGlobalParametrization)
add_subdirectory(TFHEToConcrete)
add_subdirectory(FHETensorOpsToLinalg)
add_subdirectory(ConcreteToBConcrete)
add_subdirectory(BConcreteToBConcreteCAPI)
add_subdirectory(MLIRLowerableDialectsToLLVM)
add_subdirectory(LinalgExtras)

add_mlir_library(ConcretelangConversion
  Tools.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
