// RUN: concretecompiler --split-input-file --action=dump-parametrized-tfhe --skip-program-info --passes=tfhe-circuit-solution-parametrization %s 2>&1| FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   func.func @main(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:     %0 = call @aux(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     return %0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @aux(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:     %0 = "TFHE.add_glwe"(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     return %0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @main(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>) {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a1 = "TypeInference.propagate_downward"(%arg1) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %0 = call @aux(%a0, %a1) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  return %0: !TFHE.glwe<sk?>
}

func.func @aux(%arg0: !TFHE.glwe<sk?>, %arg1: !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>) {
  %0 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  return %0: !TFHE.glwe<sk?>
}

// -----

// CHECK:      module {
// CHECK-NEXT:   func.func @main(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:     %0 = call @intermediate(%arg0, %arg1, %arg0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     return %0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @intermediate(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>, %arg2: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:     %0 = call @aux(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     %1 = "TFHE.add_glwe"(%arg2, %0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     return %1 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @aux(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:     %0 = "TFHE.add_glwe"(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     return %0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @main(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>) {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %0 = call @intermediate(%a0, %arg1, %arg0) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>, !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  return %0: !TFHE.glwe<sk?>
}

func.func @intermediate(%arg0: !TFHE.glwe<sk?>, %arg1: !TFHE.glwe<sk?>, %arg2: !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>) {
  %a2 = "TypeInference.propagate_downward"(%arg2) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %0 = call @aux(%arg0, %arg1) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  %1 = "TFHE.add_glwe"(%a2, %0): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  return %1: !TFHE.glwe<sk?>
}

func.func @aux(%arg0: !TFHE.glwe<sk?>, %arg1: !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>) {
  %0 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  return %0: !TFHE.glwe<sk?>
}
