// RUN: concretecompiler --passes fhe-to-tfhe-crt --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

// CHECK:  func.func @sub_int_eint(%[[Varg0:.*]]: tensor<5x!TFHE.glwe<sk?>>) -> tensor<5x!TFHE.glwe<sk?>> {
// CHECK-NEXT:    %[[Vc1_i8:.*]] = arith.constant 1 : i8
// CHECK-NEXT:    %[[V0:.*]] = arith.extsi %[[Vc1_i8]] : i8 to i64
// CHECK-NEXT:    %[[V1:.*]] = "TFHE.encode_plaintext_with_crt"(%[[V0]]) {mods = [2, 3, 5, 7, 11], modsProd = 2310 : i64} : (i64) -> tensor<5xi64>
// CHECK-NEXT:    %[[V2:.*]] = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:    %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[Vc1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[Vc5:.*]] = arith.constant 5 : index
// CHECK-NEXT:    %[[V3:.*]] = scf.for %[[Varg1:.*]] = %[[Vc0]] to %[[Vc5]] step %[[Vc1]] iter_args(%[[Varg2:.*]] = %[[V2]]) -> (tensor<5x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:      %[[V4:.*]] = tensor.extract %[[Varg0]][%[[Varg1]]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:      %[[V5:.*]] = tensor.extract %[[V1]][%[[Varg1]]] : tensor<5xi64>
// CHECK-NEXT:      %[[V6:.*]] = "TFHE.sub_int_glwe"(%[[V5]], %[[V4]]) : (i64, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
// CHECK-NEXT:      %[[V7:.*]] = tensor.insert %[[V6]] into %[[Varg2]][%[[Varg1]]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:      scf.yield %[[V7]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V3]] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:  }
func.func @sub_int_eint(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %0 = arith.constant 1 : i8
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i8, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}

