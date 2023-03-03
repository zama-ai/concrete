// RUN: concretecompiler --action=dump-llvm-ir %s
// Just ensure that compile
// https://github.com/zama-ai/concrete-compiler-internal/issues/785
func.func @main(%arg0: !FHE.eint<15>, %cst: tensor<32768xi64>) -> tensor<1x!FHE.eint<15>> {
  %1 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<15>, tensor<32768xi64>) -> !FHE.eint<15>
  %6 = tensor.from_elements %1 : tensor<1x!FHE.eint<15>>   // ERROR HERE line 4
  return %6 : tensor<1x!FHE.eint<15>>
}

// Ensures that tensors of multiple elements can be constructed as well.
func.func @main2(%arg0: !FHE.eint<15>, %cst: tensor<32768xi64>) -> tensor<2x!FHE.eint<15>> {
  %1 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<15>, tensor<32768xi64>) -> !FHE.eint<15>
  %6 = tensor.from_elements %1, %arg0 : tensor<2x!FHE.eint<15>>   // ERROR HERE line 4
  return %6 : tensor<2x!FHE.eint<15>>
}
