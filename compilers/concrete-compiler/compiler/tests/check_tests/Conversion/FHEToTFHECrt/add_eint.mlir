// RUN: concretecompiler --optimize-tfhe=false --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

//CHECK:  func.func @add_eint(%[[Varg0:.*]]: tensor<5x!TFHE.glwe<{_,_,_}{7}>>, %[[Varg1:.*]]: tensor<5x!TFHE.glwe<{_,_,_}{7}>>) -> tensor<5x!TFHE.glwe<{_,_,_}{7}>> {
//CHECK-NEXT:    %[[V0:.*]] = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:    %[[Vc0:.*]] = arith.constant 0 : index
//CHECK-NEXT:    %[[Vc1:.*]] = arith.constant 1 : index
//CHECK-NEXT:    %[[Vc5:.*]] = arith.constant 5 : index
//CHECK-NEXT:    %[[V1:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc5]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V0]]) -> (tensor<5x!TFHE.glwe<{_,_,_}{7}>>) {
//CHECK-NEXT:      %[[V2:.*]] = tensor.extract %[[Varg0]][%[[Varg2]]] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:      %[[V3:.*]] = tensor.extract %[[Varg1]][%[[Varg2]]] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:      %[[V4:.*]] = "TFHE.add_glwe"(%[[V2]], %[[V3]]) : (!TFHE.glwe<{_,_,_}{7}>, !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
//CHECK-NEXT:      %[[V5:.*]] = tensor.insert %[[V4]] into %[[Varg3]][%[[Varg2]]] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:      scf.yield %[[V5]] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:    }
//CHECK-NEXT:    return %[[V1]] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
func.func @add_eint(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
