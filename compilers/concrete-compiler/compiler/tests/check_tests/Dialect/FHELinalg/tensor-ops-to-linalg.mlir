// RUN: concretecompiler %s --action=dump-fhe-no-linalg 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   func.func @dot_eint_int(%[[Varg0:.*]]: tensor<2x!FHE.eint<2>>, %[[Varg1:.*]]: tensor<2xi3>) -> !FHE.eint<2> {
// CHECK-NEXT:     %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[Vc2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[Vc1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<2>>
// CHECK-NEXT:     %[[V1:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V0]]) -> (tensor<1x!FHE.eint<2>>) {
// CHECK-NEXT:       %[[V3:.*]] = tensor.extract %[[Varg0]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!FHE.eint<2>>
// CHECK-NEXT:       %[[V4:.*]] = tensor.extract %[[Varg1]]{{\[}}%[[Varg2]]{{\]}} : tensor<2xi3>
// CHECK-NEXT:       %[[V5:.*]] = tensor.extract %[[Varg3]]{{\[}}%[[Vc0]]{{\]}} : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:       %[[V6:.*]] = "FHE.mul_eint_int"(%[[V3]], %[[V4]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
// CHECK-NEXT:       %[[V7:.*]] = "FHE.add_eint"(%[[V6]], %[[V5]]) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
// CHECK-NEXT:       %[[V8:.*]] = tensor.insert %[[V7]] into %[[Varg3]]{{\[}}%[[Vc0]]{{\]}} : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:       scf.yield %[[V8]] : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V2:.*]] = tensor.extract %[[V1]]{{\[}}%[[Vc0]]{{\]}} : tensor<1x!FHE.eint<2>>
// CHECK-NEXT:     return %[[V2]] : !FHE.eint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @dot_eint_int(%arg0: tensor<2x!FHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !FHE.eint<2>
{
  %o = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!FHE.eint<2>>, tensor<2xi3>) -> !FHE.eint<2>
  return %o : !FHE.eint<2>
}
