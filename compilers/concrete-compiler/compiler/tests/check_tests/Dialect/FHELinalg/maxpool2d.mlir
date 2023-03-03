// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHELinalg.maxpool2d"(%[[a0]]) {kernel_shape = dense<[4, 2]> : tensor<2xi64>} : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<1x1x13x9x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>> {
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[4, 2]> : tensor<2xi64> } : (tensor<1x1x16x10x!FHE.eint<7>>) -> tensor<1x1x13x9x!FHE.eint<7>>
  return %0 : tensor<1x1x13x9x!FHE.eint<7>>
}
