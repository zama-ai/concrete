// RUN: concretecompiler --optimize-tfhe=false --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

//CHECK: func.func @conv2d(%[[Varg0:.*]]: tensor<100x3x28x28x5x!TFHE.glwe<{_,_,_}{2}>>, %[[Varg1:.*]]: tensor<4x3x14x14xi3>, %[[Varg2:.*]]: tensor<4xi3>) -> tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>> {
//CHECK-NEXT:    %[[Vc0:.*]] = arith.constant 0 : index
//CHECK-NEXT:    %[[Vc100:.*]] = arith.constant 100 : index
//CHECK-NEXT:    %[[Vc1:.*]] = arith.constant 1 : index
//CHECK-NEXT:    %[[Vc4:.*]] = arith.constant 4 : index
//CHECK-NEXT:    %[[Vc15:.*]] = arith.constant 15 : index
//CHECK-NEXT:    %[[Vc3:.*]] = arith.constant 3 : index
//CHECK-NEXT:    %[[Vc14:.*]] = arith.constant 14 : index
//CHECK-NEXT:    %[[V0:.*]] = "TFHE.zero_tensor"() : () -> tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:    %[[V1:.*]] = scf.for %[[Varg3:.*]] = %[[Vc0]] to %[[Vc100]] step %[[Vc1]] iter_args(%[[Varg4:.*]] = %[[V0]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:      %[[V3:.*]] = scf.for %[[Varg5:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc1]] iter_args(%[[Varg6:.*]] = %[[Varg4]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:        %[[V4:.*]] = scf.for %[[Varg7:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg8:.*]] = %[[Varg6]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:          %[[V5:.*]] = scf.for %[[Varg9:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg10:.*]] = %[[Varg8]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:            %[[Vextracted:.*]] = tensor.extract %[[Varg2]]{{\[}}%[[Varg5]]{{\]}} : tensor<4xi3>
//CHECK-NEXT:            %[[Vc0_0:.*]] = arith.constant 0 : index
//CHECK-NEXT:            %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg10]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]], %[[Vc0_0]]{{\] \[1, 1, 1, 1, 5\] \[1, 1, 1, 1, 1\]}} : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>> to tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            %[[V6:.*]] = arith.extsi %[[Vextracted]] : i3 to i64
//CHECK-NEXT:            %[[V7:.*]] = "TFHE.encode_plaintext_with_crt"(%[[V6]]) {mods = {{\[2, 3, 5, 7, 11\], modsProd}} = 2310 : i64} : (i64) -> tensor<5xi64>
//CHECK-NEXT:            %[[V8:.*]] = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            %[[Vc0_1:.*]] = arith.constant 0 : index
//CHECK-NEXT:            %[[Vc1_2:.*]] = arith.constant 1 : index
//CHECK-NEXT:            %[[Vc5:.*]] = arith.constant 5 : index
//CHECK-NEXT:            %[[V9:.*]] = scf.for %[[Varg11:.*]] = %[[Vc0_1]] to %[[Vc5]] step %[[Vc1_2]] iter_args(%[[Varg12:.*]] = %[[V8]]) -> (tensor<5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:              %[[Vextracted_4:.*]] = tensor.extract %[[Vextracted_slice]]{{\[}}%[[Varg11]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:              %[[Vextracted_5:.*]] = tensor.extract %[[V7]]{{\[}}%[[Varg11]]{{\]}} : tensor<5xi64>
//CHECK-NEXT:              %[[V10:.*]] = "TFHE.add_glwe_int"(%[[Vextracted_4]], %[[Vextracted_5]]) : (!TFHE.glwe<{_,_,_}{2}>, i64) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:              %[[Vinserted:.*]] = tensor.insert %[[V10]] into %[[Varg12]]{{\[}}%[[Varg11]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:              scf.yield %[[Vinserted]] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            }
//CHECK-NEXT:            %[[Vc0_3:.*]] = arith.constant 0 : index
//CHECK-NEXT:            %[[Vinserted_slice:.*]] = tensor.insert_slice %[[V9]] into %[[Varg10]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]], %[[Vc0_3]]{{\] \[1, 1, 1, 1, 5\] \[1, 1, 1, 1, 1\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>> into tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            scf.yield %[[Vinserted_slice]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:          }
//CHECK-NEXT:          scf.yield %[[V5]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:        }
//CHECK-NEXT:        scf.yield %[[V4]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:      }
//CHECK-NEXT:      scf.yield %[[V3]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:    }
//CHECK-NEXT:    %[[V2:.*]] = scf.for %[[Varg3:.*]] = %[[Vc0]] to %[[Vc100]] step %[[Vc1]] iter_args(%[[Varg4:.*]] = %[[V1]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:      %[[V3:.*]] = scf.for %[[Varg5:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc1]] iter_args(%[[Varg6:.*]] = %[[Varg4]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:        %[[V4:.*]] = scf.for %[[Varg7:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg8:.*]] = %[[Varg6]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:          %[[V5:.*]] = scf.for %[[Varg9:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg10:.*]] = %[[Varg8]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:            %[[V6:.*]] = scf.for %[[Varg11:.*]] = %[[Vc0]] to %[[Vc3]] step %[[Vc1]] iter_args(%[[Varg12:.*]] = %[[Varg10]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:              %[[V7:.*]] = scf.for %[[Varg13:.*]] = %[[Vc0]] to %[[Vc14]] step %[[Vc1]] iter_args(%[[Varg14:.*]] = %[[Varg12]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                %[[V8:.*]] = scf.for %[[Varg15:.*]] = %[[Vc0]] to %[[Vc14]] step %[[Vc1]] iter_args(%[[Varg16:.*]] = %[[Varg14]]) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                  %[[V9:.*]] = affine.apply #map(%[[Varg7]], %[[Varg13]])
//CHECK-NEXT:                  %[[V10:.*]] = affine.apply #map(%[[Varg9]], %[[Varg15]])
//CHECK-NEXT:                  %[[Vc0_0:.*]] = arith.constant 0 : index
//CHECK-NEXT:                  %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[Varg3]], %[[Varg11]], %[[V9]], %[[V10]], %[[Vc0_0]]{{\] \[1, 1, 1, 1, 5\] \[1, 1, 1, 1, 1\]}} : tensor<100x3x28x28x5x!TFHE.glwe<{_,_,_}{2}>> to tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %[[Vextracted:.*]] = tensor.extract %[[Varg1]]{{\[}}%[[Varg5]], %[[Varg11]], %[[Varg13]], %[[Varg15]]{{\]}} : tensor<4x3x14x14xi3>
//CHECK-NEXT:                  %[[Vc0_1:.*]] = arith.constant 0 : index
//CHECK-NEXT:                  %[[Vextracted_slice_2:.*]] = tensor.extract_slice %[[Varg16]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]], %[[Vc0_1]]{{\] \[1, 1, 1, 1, 5\] \[1, 1, 1, 1, 1\]}} : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>> to tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %[[V11:.*]] = arith.extsi %[[Vextracted]] : i3 to i64
//CHECK-NEXT:                  %[[V12:.*]] = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %[[Vc0_3:.*]] = arith.constant 0 : index
//CHECK-NEXT:                  %[[Vc1_4:.*]] = arith.constant 1 : index
//CHECK-NEXT:                  %[[Vc5:.*]] = arith.constant 5 : index
//CHECK-NEXT:                  %[[V13:.*]] = scf.for %[[Varg17:.*]] = %[[Vc0_3]] to %[[Vc5]] step %[[Vc1_4]] iter_args(%[[Varg18:.*]] = %[[V12]]) -> (tensor<5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                    %[[Vextracted_9:.*]] = tensor.extract %[[Vextracted_slice]]{{\[}}%[[Varg17]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    %[[V16:.*]] = "TFHE.mul_glwe_int"(%[[Vextracted_9]], %[[V11]]) : (!TFHE.glwe<{_,_,_}{2}>, i64) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:                    %[[Vinserted:.*]] = tensor.insert %[[V16]] into %[[Varg18]]{{\[}}%[[Varg17]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    scf.yield %[[Vinserted]] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  }
//CHECK-NEXT:                  %[[V14:.*]] = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %[[Vc0_5:.*]] = arith.constant 0 : index
//CHECK-NEXT:                  %[[Vc1_6:.*]] = arith.constant 1 : index
//CHECK-NEXT:                  %[[Vc5_7:.*]] = arith.constant 5 : index
//CHECK-NEXT:                  %[[V15:.*]] = scf.for %[[Varg17:.*]] = %[[Vc0_5]] to %[[Vc5_7]] step %[[Vc1_6]] iter_args(%[[Varg18:.*]] = %[[V14]]) -> (tensor<5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                    %[[Vextracted_9:.*]] = tensor.extract %[[Vextracted_slice_2]]{{\[}}%[[Varg17]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    %[[Vextracted_10:.*]] = tensor.extract %[[V13]]{{\[}}%[[Varg17]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    %[[V16:.*]] = "TFHE.add_glwe"(%[[Vextracted_9]], %[[Vextracted_10]]) : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:                    %[[Vinserted:.*]] = tensor.insert %[[V16]] into %[[Varg18]]{{\[}}%[[Varg17]]{{\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    scf.yield %[[Vinserted]] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  }
//CHECK-NEXT:                  %[[Vc0_8:.*]] = arith.constant 0 : index
//CHECK-NEXT:                  %[[Vinserted_slice:.*]] = tensor.insert_slice %[[V15]] into %[[Varg16]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]], %[[Vc0_8]]{{\] \[1, 1, 1, 1, 5\] \[1, 1, 1, 1, 1\]}} : tensor<5x!TFHE.glwe<{_,_,_}{2}>> into tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  scf.yield %[[Vinserted_slice]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                }
//CHECK-NEXT:                scf.yield %[[V8]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:              }
//CHECK-NEXT:              scf.yield %[[V7]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            }
//CHECK-NEXT:            scf.yield %[[V6]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:          }
//CHECK-NEXT:          scf.yield %[[V5]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:        }
//CHECK-NEXT:        scf.yield %[[V4]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:      }
//CHECK-NEXT:      scf.yield %[[V3]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:    }
//CHECK-NEXT:    return %[[V2]] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:  }
func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0, 0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}
