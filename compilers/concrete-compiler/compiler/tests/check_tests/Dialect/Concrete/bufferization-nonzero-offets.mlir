// RUN: concretecompiler --force-encoding native -action=dump-llvm-dialect %s
// see https://github.com/zama-ai/concrete-compiler-internal/issues/858 for encoding crt

// Extracted from the source referenced in Issue 663. This should
// trigger the folding of memrefs of itermediate results to memrefs
// with non-zero offsets. Prior to the use of symbolic offsets in the
// memref used in the memref.cast operation produced by the Concrete
// bufferizer, bufferization of the function below would fail.
func.func @main(%arg0: tensor<32x!FHE.eint<8>>, %arg1: tensor<256xi64>) -> !FHE.eint<8>
{
  %c0 = arith.constant 0 : index
  %719 = "FHELinalg.apply_lookup_table"(%arg0, %arg1) : (tensor<32x!FHE.eint<8>>, tensor<256xi64>) -> tensor<32x!FHE.eint<8>>
  %720 = tensor.extract_slice %719[16] [16] [1] : tensor<32x!FHE.eint<8>> to tensor<16x!FHE.eint<8>>
  %722 = tensor.extract %720[%c0] : tensor<16x!FHE.eint<8>>
  %755 = "FHE.add_eint"(%722, %722) : (!FHE.eint<8>, !FHE.eint<8>) -> !FHE.eint<8>
  return %755 : !FHE.eint<8>
}
