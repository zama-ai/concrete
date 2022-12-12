// RUN: concretecompiler --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

//CHECK-LABEL:  func.func @add_eint(%arg0: tensor<5x!TFHE.glwe<{_,_,_}{7}>>, %arg1: tensor<5x!TFHE.glwe<{_,_,_}{7}>>) -> tensor<5x!TFHE.glwe<{_,_,_}{7}>> {
//CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    %c5 = arith.constant 5 : index
//CHECK-NEXT:    %1 = scf.for %arg2 = %c0 to %c5 step %c1 iter_args(%arg3 = %0) -> (tensor<5x!TFHE.glwe<{_,_,_}{7}>>) {
//CHECK-NEXT:      %2 = tensor.extract %arg0[%arg2] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:      %3 = tensor.extract %arg1[%arg2] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:      %4 = "TFHE.add_glwe"(%2, %3) : (!TFHE.glwe<{_,_,_}{7}>, !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
//CHECK-NEXT:      %5 = tensor.insert %4 into %arg3[%arg2] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:      scf.yield %5 : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
//CHECK-NEXT:    }
//CHECK-NEXT:    return %1 : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
func.func @add_eint(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
