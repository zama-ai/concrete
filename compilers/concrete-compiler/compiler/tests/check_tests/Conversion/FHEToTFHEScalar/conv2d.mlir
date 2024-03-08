// RUN: concretecompiler %s --optimize-tfhe=false --optimizer-strategy=dag-mono --action=dump-tfhe 2>&1| FileCheck %s

// CHECK: func.func @conv2d(%[[Varg0:.*]]: tensor<100x3x28x28x!TFHE.glwe<sk?>>, %[[Varg1:.*]]: tensor<4x3x14x14xi3>, %[[Varg2:.*]]: tensor<4xi3>) -> tensor<100x4x15x15x!TFHE.glwe<sk?>> {
// CHECK-NEXT:    %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[Vc100:.*]] = arith.constant 100 : index
// CHECK-NEXT:    %[[Vc1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[Vc4:.*]] = arith.constant 4 : index
// CHECK-NEXT:    %[[Vc15:.*]] = arith.constant 15 : index
// CHECK-NEXT:    %[[Vc3:.*]] = arith.constant 3 : index
// CHECK-NEXT:    %[[Vc14:.*]] = arith.constant 14 : index
// CHECK-NEXT:    %[[V0:.*]] = "TFHE.zero_tensor"() : () -> tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:    %[[V1:.*]] = scf.for %[[Varg3:.*]] = %[[Vc0]] to %[[Vc100]] step %[[Vc1]] iter_args(%[[Varg4:.*]] = %[[V0]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:      %[[V3:.*]] = scf.for %[[Varg5:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc1]] iter_args(%[[Varg6:.*]] = %[[Varg4]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:        %[[V4:.*]] = scf.for %[[Varg7:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg8:.*]] = %[[Varg6]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:          %[[V5:.*]] = scf.for %[[Varg9:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg10:.*]] = %[[Varg8]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:            %[[Vextracted:.*]] = tensor.extract %[[Varg2]]{{\[}}%[[Varg5]]{{\]}} : tensor<4xi3>
// CHECK-NEXT:            %[[Vextracted_0:.*]] = tensor.extract %[[Varg10]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]]{{\]}} : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:            %[[V6:.*]] = arith.extsi %[[Vextracted]] : i3 to i64
// CHECK-NEXT:            %[[Vc61_i64:.*]] = arith.constant 61 : i64
// CHECK-NEXT:            %[[V7:.*]] = arith.shli %[[V6]], %[[Vc61_i64]] : i64
// CHECK-NEXT:            %[[V8:.*]] = "TFHE.add_glwe_int"(%[[Vextracted_0]], %[[V7]]) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
// CHECK-NEXT:            %[[Vinserted:.*]] = tensor.insert %[[V8]] into %[[Varg10]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]]{{\]}} : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:            scf.yield %[[Vinserted]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %[[V5]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[V4]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V3]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V2:.*]] = scf.for %[[Varg3:.*]] = %[[Vc0]] to %[[Vc100]] step %[[Vc1]] iter_args(%[[Varg4:.*]] = %[[V1]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:      %[[V3:.*]] = scf.for %[[Varg5:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc1]] iter_args(%[[Varg6:.*]] = %[[Varg4]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:        %[[V4:.*]] = scf.for %[[Varg7:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg8:.*]] = %[[Varg6]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:          %[[V5:.*]] = scf.for %[[Varg9:.*]] = %[[Vc0]] to %[[Vc15]] step %[[Vc1]] iter_args(%[[Varg10:.*]] = %[[Varg8]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:            %[[V6:.*]] = scf.for %[[Varg11:.*]] = %[[Vc0]] to %[[Vc3]] step %[[Vc1]] iter_args(%[[Varg12:.*]] = %[[Varg10]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:              %[[V7:.*]] = scf.for %[[Varg13:.*]] = %[[Vc0]] to %[[Vc14]] step %[[Vc1]] iter_args(%[[Varg14:.*]] = %[[Varg12]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:                %[[V8:.*]] = scf.for %[[Varg15:.*]] = %[[Vc0]] to %[[Vc14]] step %[[Vc1]] iter_args(%[[Varg16:.*]] = %[[Varg14]]) -> (tensor<100x4x15x15x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:                  %[[V9:.*]] = affine.apply #map(%[[Varg7]], %[[Varg13]])
// CHECK-NEXT:                  %[[V10:.*]] = affine.apply #map(%[[Varg9]], %[[Varg15]])
// CHECK-NEXT:                  %[[Vextracted:.*]] = tensor.extract %[[Varg0]]{{\[}}%[[Varg3]], %[[Varg11]], %[[V9]], %[[V10]]{{\]}} : tensor<100x3x28x28x!TFHE.glwe<sk?>>
// CHECK-NEXT:                  %[[Vextracted_0:.*]] = tensor.extract %[[Varg1]]{{\[}}%[[Varg5]], %[[Varg11]], %[[Varg13]], %[[Varg15]]{{\]}} : tensor<4x3x14x14xi3>
// CHECK-NEXT:                  %[[Vextracted_1:.*]] = tensor.extract %[[Varg16]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]]{{\]}} : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:                  %[[V11:.*]] = arith.extsi %[[Vextracted_0]] : i3 to i64
// CHECK-NEXT:                  %[[V12:.*]] = "TFHE.mul_glwe_int"(%[[Vextracted]], %[[V11]]) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
// CHECK-NEXT:                  %[[V13:.*]] = "TFHE.add_glwe"(%[[Vextracted_1]], %[[V12]]) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
// CHECK-NEXT:                  %[[Vinserted:.*]] = tensor.insert %[[V13]] into %[[Varg16]]{{\[}}%[[Varg3]], %[[Varg5]], %[[Varg7]], %[[Varg9]]{{\]}} : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:                  scf.yield %[[Vinserted]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield %[[V8]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.yield %[[V7]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[V6]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %[[V5]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[V4]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V3]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V2]] : tensor<100x4x15x15x!TFHE.glwe<sk?>>
// CHECK-NEXT:   }
func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0, 0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}
