// RUN: concretecompiler %s --action=dump-fhe-no-linalg 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   func.func @dot_eint_int(%arg0: tensor<2x!FHE.eint<2>>, %arg1: tensor<2xi3>) -> !FHE.eint<2> {
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<2>>
// CHECK-NEXT:     %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<1x!FHE.eint<2>>) {
// CHECK-NEXT:       %3 = tensor.extract %arg0[%arg2] : tensor<2x!FHE.eint<2>>
// CHECK-NEXT:       %4 = tensor.extract %arg1[%arg2] : tensor<2xi3>
// CHECK-NEXT:       %5 = tensor.extract %0[%c0] : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:       %6 = "FHE.mul_eint_int"(%3, %4) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
// CHECK-NEXT:       %7 = "FHE.add_eint"(%6, %5) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
// CHECK-NEXT:       %8 = tensor.insert %7 into %arg3[%c0] : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:       scf.yield %8 : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = tensor.extract %1[%c0] : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:     return %2 : !FHE.eint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @dot_eint_int(%arg0: tensor<2x!FHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !FHE.eint<2>
{
  %o = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!FHE.eint<2>>, tensor<2xi3>) -> !FHE.eint<2>
  return %o : !FHE.eint<2>
}
