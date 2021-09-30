// RUN: zamacompiler --entry-dialect=midlfhe --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: tensor<128xi64>) -> !MidLFHE.glwe<{512,10,64}{2}>
func @apply_lookup_table(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: tensor<128xi64>) -> !MidLFHE.glwe<{512,10,64}{2}> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.apply_lookup_table"(%arg0, %arg1) {baseLogBS = -83 : i32, baseLogKS = -82 : i32, k = 1 : i32, levelBS = 3 : i32, levelKS = 2 : i32, outputSizeKS = 600 : i32, polynomialSize = 1024 : i32} : (!MidLFHE.glwe<{1024,12,64}{7}>, tensor<128xi64>) -> !MidLFHE.glwe<{512,10,64}{2}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{512,10,64}{2}>

  %1 = "MidLFHE.apply_lookup_table"(%arg0, %arg1) {k = 1 : i32, polynomialSize = 1024 : i32, levelKS = 2 : i32, baseLogKS = -82 : i32, levelBS = 3 : i32, baseLogBS = -83 : i32, outputSizeKS = 600 : i32} : (!MidLFHE.glwe<{1024,12,64}{7}>, tensor<128xi64>) -> (!MidLFHE.glwe<{512,10,64}{2}>)
  return %1: !MidLFHE.glwe<{512,10,64}{2}>
}
