// RUN: concretecompiler --action=dump-parametrized-tfhe --optimizer-strategy=dag-multi %s

// CHECK: module {
// CHECK-NEXT:   func.func @main(%[[Varg0:.*]]: tensor<2x!TFHE.glwe<sk<0,1,1536>>>, %[[Varg1:.*]]: tensor<2x!TFHE.glwe<sk<1,1,8192>>>) -> (tensor<2x!TFHE.glwe<sk<0,1,1536>>>, tensor<2x!TFHE.glwe<sk<1,1,8192>>>) {
// CHECK-NEXT:     %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[Vc2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[Vc1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[Vcst:.*]] = arith.constant dense<0> : tensor<128xi64>
// CHECK-NEXT:     %[[Vcst_0:.*]] = arith.constant dense<{{\[0, 1\]}}> : tensor<2xi64>
// CHECK-NEXT:     %[[V0:.*]] = "TFHE.zero_tensor"() : () -> tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:     %[[V1:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V0]]) -> (tensor<2x!TFHE.glwe<sk<0,1,1536>>>) {
// CHECK-NEXT:       %[[Vextracted:.*]] = tensor.extract %[[Varg0]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:       %[[V10:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Vcst_0]]) {isSigned = false, outputBits = 8 : i32, polySize = 256 : i32} : (tensor<2xi64>) -> tensor<256xi64>
// CHECK-NEXT:       %[[V11:.*]] = "TFHE.keyswitch_glwe"(%[[Vextracted]]) {key = #TFHE.ksk<sk<0,1,1536>, sk<2,1,604>, 3, 3>} : (!TFHE.glwe<sk<0,1,1536>>) -> !TFHE.glwe<sk<2,1,604>>
// CHECK-NEXT:       %[[V12:.*]] = "TFHE.bootstrap_glwe"(%[[V11]], %[[V10]]) {key = #TFHE.bsk<sk<2,1,604>, sk<0,1,1536>, 256, 6, 2, 12>} : (!TFHE.glwe<sk<2,1,604>>, tensor<256xi64>) -> !TFHE.glwe<sk<0,1,1536>>
// CHECK-NEXT:       %[[Vinserted:.*]] = tensor.insert %[[V12]] into %[[Varg3]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:       scf.yield %[[Vinserted]] : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V2:.*]] = "TFHE.zero_tensor"() : () -> tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:     %[[V3:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V2]]) -> (tensor<2x!TFHE.glwe<sk<1,1,8192>>>) {
// CHECK-NEXT:       %[[Vextracted:.*]] = tensor.extract %[[Varg1]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:       %[[V10:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Vcst]]) {isSigned = false, outputBits = 8 : i32, polySize = 8192 : i32} : (tensor<128xi64>) -> tensor<8192xi64>
// CHECK-NEXT:       %[[V11:.*]] = "TFHE.keyswitch_glwe"(%[[Vextracted]]) {key = #TFHE.ksk<sk<1,1,8192>, sk<3,1,942>, 6, 3>} : (!TFHE.glwe<sk<1,1,8192>>) -> !TFHE.glwe<sk<3,1,942>>
// CHECK-NEXT:       %[[V12:.*]] = "TFHE.bootstrap_glwe"(%[[V11]], %[[V10]]) {key = #TFHE.bsk<sk<3,1,942>, sk<1,1,8192>, 8192, 1, 1, 22>} : (!TFHE.glwe<sk<3,1,942>>, tensor<8192xi64>) -> !TFHE.glwe<sk<1,1,8192>>
// CHECK-NEXT:       %[[Vinserted:.*]] = tensor.insert %[[V12]] into %[[Varg3]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:       scf.yield %[[Vinserted]] : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V4:.*]] = tensor.empty() : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:     %[[V5:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V4]]) -> (tensor<2x!TFHE.glwe<sk<0,1,1536>>>) {
// CHECK-NEXT:       %[[Vextracted:.*]] = tensor.extract %[[V3]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:       %[[V10:.*]] = "TFHE.keyswitch_glwe"(%[[Vextracted]]) {key = #TFHE.ksk<sk<1,1,8192>, sk<0,1,1536>, 1, 19>} : (!TFHE.glwe<sk<1,1,8192>>) -> !TFHE.glwe<sk<0,1,1536>>
// CHECK-NEXT:       %[[Vinserted:.*]] = tensor.insert %[[V10]] into %[[Varg3]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:       scf.yield %[[Vinserted]] : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V6:.*]] = "TFHE.zero_tensor"() : () -> tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:     %[[V7:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V6]]) -> (tensor<2x!TFHE.glwe<sk<0,1,1536>>>) {
// CHECK-NEXT:       %[[Vextracted:.*]] = tensor.extract %[[V1]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:       %[[Vextracted_1:.*]] = tensor.extract %[[V5]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:       %[[V10:.*]] = "TFHE.add_glwe"(%[[Vextracted]], %[[Vextracted_1]]) : (!TFHE.glwe<sk<0,1,1536>>, !TFHE.glwe<sk<0,1,1536>>) -> !TFHE.glwe<sk<0,1,1536>>
// CHECK-NEXT:       %[[Vinserted:.*]] = tensor.insert %[[V10]] into %[[Varg3]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:       scf.yield %[[Vinserted]] : tensor<2x!TFHE.glwe<sk<0,1,1536>>>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V8:.*]] = "TFHE.zero_tensor"() : () -> tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:     %[[V9:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg3:.*]] = %[[V8]]) -> (tensor<2x!TFHE.glwe<sk<1,1,8192>>>) {
// CHECK-NEXT:       %[[Vextracted:.*]] = tensor.extract %[[Varg1]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:       %[[V10:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Vcst]]) {isSigned = false, outputBits = 5 : i32, polySize = 8192 : i32} : (tensor<128xi64>) -> tensor<8192xi64>
// CHECK-NEXT:       %[[V11:.*]] = "TFHE.keyswitch_glwe"(%[[Vextracted]]) {key = #TFHE.ksk<sk<1,1,8192>, sk<3,1,942>, 6, 3>} : (!TFHE.glwe<sk<1,1,8192>>) -> !TFHE.glwe<sk<3,1,942>>
// CHECK-NEXT:       %[[V12:.*]] = "TFHE.bootstrap_glwe"(%[[V11]], %[[V10]]) {key = #TFHE.bsk<sk<3,1,942>, sk<1,1,8192>, 8192, 1, 1, 22>} : (!TFHE.glwe<sk<3,1,942>>, tensor<8192xi64>) -> !TFHE.glwe<sk<1,1,8192>>
// CHECK-NEXT:       %[[Vinserted:.*]] = tensor.insert %[[V12]] into %[[Varg3]]{{\[}}%[[Varg2]]{{\]}} : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:       scf.yield %[[Vinserted]] : tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V7]], %[[V9]] : tensor<2x!TFHE.glwe<sk<0,1,1536>>>, tensor<2x!TFHE.glwe<sk<1,1,8192>>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @main(%arg0: tensor<2x!FHE.eint<1>>, %arg1: tensor<2x!FHE.eint<7>>) -> (tensor<2x!FHE.eint<8>>, tensor<2x!FHE.eint<5>>) {
  %cst = arith.constant dense<[0, 1]> : tensor<2xi64>
  %0 = "FHELinalg.apply_lookup_table"(%arg0, %cst) : (tensor<2x!FHE.eint<1>>, tensor<2xi64>) -> tensor<2x!FHE.eint<8>>
  %cst_0 = arith.constant dense<0> : tensor<128xi64>
  %1 = "FHELinalg.apply_lookup_table"(%arg1, %cst_0) : (tensor<2x!FHE.eint<7>>, tensor<128xi64>) -> tensor<2x!FHE.eint<8>>
  %2 = "FHELinalg.add_eint"(%0, %1) : (tensor<2x!FHE.eint<8>>, tensor<2x!FHE.eint<8>>) -> tensor<2x!FHE.eint<8>>
  %c4_i4 = arith.constant 4 : i4
  %cst_1 = arith.constant dense<0> : tensor<256xi64>
  %3 = "FHELinalg.apply_lookup_table"(%1, %cst_1) : (tensor<2x!FHE.eint<8>>, tensor<256xi64>) -> tensor<2x!FHE.eint<5>>
  return %2, %3 : tensor<2x!FHE.eint<8>>, tensor<2x!FHE.eint<5>>
}
