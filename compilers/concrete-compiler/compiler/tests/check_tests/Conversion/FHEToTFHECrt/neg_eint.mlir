// RUN: concretecompiler --optimize-tfhe=false --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

// CHECK-LABEL:  func.func @neg_eint(%arg0: tensor<5x!TFHE.glwe<{_,_,_}{7}>>) -> tensor<5x!TFHE.glwe<{_,_,_}{7}>> {
// CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c5 = arith.constant 5 : index
// CHECK-NEXT:    %1 = scf.for %arg1 = %c0 to %c5 step %c1 iter_args(%arg2 = %0) -> (tensor<5x!TFHE.glwe<{_,_,_}{7}>>) {
// CHECK-NEXT:      %2 = tensor.extract %arg0[%arg1] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
// CHECK-NEXT:      %3 = "TFHE.neg_glwe"(%2) : (!TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
// CHECK-NEXT:      %4 = tensor.insert %3 into %arg2[%arg1] : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
// CHECK-NEXT:      scf.yield %4 : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : tensor<5x!TFHE.glwe<{_,_,_}{7}>>
// CHECK-NEXT:  }
func.func @neg_eint(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
