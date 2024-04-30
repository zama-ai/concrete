// RUN: concretecompiler --passes fhe-to-tfhe-crt --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

// CHECK:  func.func @mul_eint_int(%[[Varg0:.*]]: tensor<5x!TFHE.glwe<sk?>>) -> tensor<5x!TFHE.glwe<sk?>> {
// CHECK-NEXT:    %[[Vc2_i8:.*]] = arith.constant 2 : i8
// CHECK-NEXT:    %[[V0:.*]] = arith.extsi %[[Vc2_i8]] : i8 to i64
// CHECK-NEXT:    %[[V1:.*]] = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:    %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[Vc1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[Vc5:.*]] = arith.constant 5 : index
// CHECK-NEXT:    %[[V2:.*]] = scf.for %[[Varg1:.*]] = %[[Vc0]] to %[[Vc5]] step %[[Vc1]] iter_args(%[[Varg2:.*]] = %[[V1]]) -> (tensor<5x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:      %[[V3:.*]] = tensor.extract %[[Varg0]][%[[Varg1]]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:      %[[V4:.*]] = "TFHE.mul_glwe_int"(%[[V3]], %[[V0]]) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
// CHECK-NEXT:      %[[V5:.*]] = tensor.insert %[[V4]] into %[[Varg2]][%[[Varg1]]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:      scf.yield %[[V5]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V2]] : tensor<5x!TFHE.glwe<sk?>>
func.func @mul_eint_int(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %0 = arith.constant 2 : i8
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
