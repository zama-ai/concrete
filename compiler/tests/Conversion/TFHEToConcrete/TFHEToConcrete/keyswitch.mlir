// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK: func @keyswitch_glwe(%[[A0:.*]]: !Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<567,2> {
// CHECK-NEXT:   %[[V0:.*]] = "Concrete.keyswitch_lwe"(%[[A0]]) {baseLog = 3 : i32, level = 2 : i32} : (!Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<567,2>
// CHECK-NEXT:   return %[[V0]] : !Concrete.lwe_ciphertext<567,2>
// CHECK-NEXT: }
func @keyswitch_glwe(%arg0: !TFHE.glwe<{1024,1,64}{2}>) -> !TFHE.glwe<{567,1,64}{2}> {
  %0 = "TFHE.keyswitch_glwe"(%arg0) {baseLog = 3 : i32, level = 2 : i32} : (!TFHE.glwe<{1024,1,64}{2}>) -> !TFHE.glwe<{567,1,64}{2}>
  return %0 : !TFHE.glwe<{567,1,64}{2}>
}
