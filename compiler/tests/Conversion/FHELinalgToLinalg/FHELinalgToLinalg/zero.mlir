// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: func @zero(%arg0: !FHE.eint<2>) -> tensor<3x2x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = tensor.generate   {
// CHECK-NEXT:   ^bb0(%arg1: index, %arg2: index):  // no predecessors
// CHECK-NEXT:     %[[yld:.*]] = "FHE.zero"() : () -> !FHE.eint<2>
// CHECK-NEXT:     tensor.yield %[[yld]] : !FHE.eint<2>
// CHECK-NEXT:   } : tensor<3x2x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x2x!FHE.eint<2>>
// CHECK-NEXT: }
func @zero(%arg0: !FHE.eint<2>) -> tensor<3x2x!FHE.eint<2>> {
  %1 = "FHELinalg.zero"(): () -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}
