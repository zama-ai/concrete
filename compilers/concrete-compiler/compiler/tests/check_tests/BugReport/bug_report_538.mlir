// RUN: concretecompiler --action=dump-parametrized-tfhe --optimizer-strategy=dag-multi %s

// CHECK-NEXT: func.func @main(%[[Varg0:.*]]: !TFHE.glwe<sk<0,1,1536>>, %[[Varg1:.*]]: !TFHE.glwe<sk<1,1,8192>>) -> (!TFHE.glwe<sk<0,1,1536>>, !TFHE.glwe<sk<2,1,16384>>) {
// CHECK-NEXT:   %[[Vcst:.*]] = arith.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:   %[[Vcst_0:.*]] = arith.constant dense<0> : tensor<128xi64>
// CHECK-NEXT:   %[[Vcst_1:.*]] = arith.constant dense<{{\[0, 1\]}}> : tensor<2xi64>
// CHECK-NEXT:   %[[V0:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Vcst_1]]) {isSigned = false, outputBits = 8 : i32, polySize = 256 : i32} : (tensor<2xi64>) -> tensor<256xi64>
// CHECK-NEXT:   %[[V1:.*]] = "TFHE.keyswitch_glwe"(%[[Varg0]]) {TFHE.OId = 2 : i32, key = #TFHE.ksk<sk<0,1,1536>, sk<3,1,601>, 3, 3>} : (!TFHE.glwe<sk<0,1,1536>>) -> !TFHE.glwe<sk<3,1,601>>
// CHECK-NEXT:   %[[V2:.*]] = "TFHE.bootstrap_glwe"(%[[V1]], %[[V0]]) {TFHE.OId = 2 : i32, key = #TFHE.bsk<sk<3,1,601>, sk<0,1,1536>, 256, 6, 2, 12>} : (!TFHE.glwe<sk<3,1,601>>, tensor<256xi64>) -> !TFHE.glwe<sk<0,1,1536>>
// CHECK-NEXT:   %[[V3:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Vcst_0]]) {isSigned = false, outputBits = 8 : i32, polySize = 8192 : i32} : (tensor<128xi64>) -> tensor<8192xi64>
// CHECK-NEXT:   %[[V4:.*]] = "TFHE.keyswitch_glwe"(%[[Varg1]]) {TFHE.OId = 3 : i32, key = #TFHE.ksk<sk<1,1,8192>, sk<4,1,923>, 6, 3>} : (!TFHE.glwe<sk<1,1,8192>>) -> !TFHE.glwe<sk<4,1,923>>
// CHECK-NEXT:   %[[V5:.*]] = "TFHE.bootstrap_glwe"(%[[V4]], %[[V3]]) {TFHE.OId = 3 : i32, key = #TFHE.bsk<sk<4,1,923>, sk<1,1,8192>, 8192, 1, 2, 15>} : (!TFHE.glwe<sk<4,1,923>>, tensor<8192xi64>) -> !TFHE.glwe<sk<1,1,8192>>
// CHECK-NEXT:   %[[V6:.*]] = "TFHE.keyswitch_glwe"(%[[V5]]) {key = #TFHE.ksk<sk<1,1,8192>, sk<0,1,1536>, 1, 19>} : (!TFHE.glwe<sk<1,1,8192>>) -> !TFHE.glwe<sk<0,1,1536>>
// CHECK-NEXT:   %[[V7:.*]] = "TFHE.add_glwe"(%[[V2]], %[[V6]]) {TFHE.OId = 4 : i32} : (!TFHE.glwe<sk<0,1,1536>>, !TFHE.glwe<sk<0,1,1536>>) -> !TFHE.glwe<sk<0,1,1536>>
// CHECK-NEXT:   %[[V8:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Vcst]]) {isSigned = false, outputBits = 5 : i32, polySize = 16384 : i32} : (tensor<256xi64>) -> tensor<16384xi64>
// CHECK-NEXT:   %[[V9:.*]] = "TFHE.keyswitch_glwe"(%[[V5]]) {TFHE.OId = 5 : i32, key = #TFHE.ksk<sk<1,1,8192>, sk<5,1,967>, 7, 3>} : (!TFHE.glwe<sk<1,1,8192>>) -> !TFHE.glwe<sk<5,1,967>>
// CHECK-NEXT:   %[[V10:.*]] = "TFHE.bootstrap_glwe"(%[[V9]], %[[V8]]) {TFHE.OId = 5 : i32, key = #TFHE.bsk<sk<5,1,967>, sk<2,1,16384>, 16384, 1, 1, 22>} : (!TFHE.glwe<sk<5,1,967>>, tensor<16384xi64>) -> !TFHE.glwe<sk<2,1,16384>>
// CHECK-NEXT:   return %[[V7]], %[[V10]] : !TFHE.glwe<sk<0,1,1536>>, !TFHE.glwe<sk<2,1,16384>>
// CHECK-NEXT: }
func.func @main(%arg0: !FHE.eint<1>, %arg1: !FHE.eint<7>) -> (!FHE.eint<8>, !FHE.eint<5>) {
  %cst = arith.constant dense<[0, 1]> : tensor<2xi64>
  %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<1>, tensor<2xi64>) -> !FHE.eint<8>
  %cst_0 = arith.constant dense<0> : tensor<128xi64>
  %1 = "FHE.apply_lookup_table"(%arg1, %cst_0) : (!FHE.eint<7>, tensor<128xi64>) -> !FHE.eint<8>
  %2 = "FHE.add_eint"(%0, %1) : (!FHE.eint<8>, !FHE.eint<8>) -> !FHE.eint<8>
  %c4_i4 = arith.constant 4 : i4
  %cst_1 = arith.constant dense<0> : tensor<256xi64>
  %3 = "FHE.apply_lookup_table"(%1, %cst_1) : (!FHE.eint<8>, tensor<256xi64>) -> !FHE.eint<5>
  return %2, %3 : !FHE.eint<8>, !FHE.eint<5>
}

