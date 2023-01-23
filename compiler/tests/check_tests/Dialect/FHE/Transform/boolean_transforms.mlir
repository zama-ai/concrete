// RUN: concretecompiler --passes fhe-boolean-transform --action=dump-fhe  %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @gen_gate(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: tensor<4xi64>) -> !FHE.ebool
func.func @gen_gate(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: tensor<4xi64>) -> !FHE.ebool {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 2 : i3
  // CHECK-NEXT: %[[V0:.*]] = "FHE.from_bool"(%arg0) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.from_bool"(%arg1) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%[[V0]], %[[C0]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint"(%[[V2]], %[[V1]]) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V4:.*]] = "FHE.apply_lookup_table"(%[[V3]], %arg2) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V5:.*]] = "FHE.to_bool"(%[[V4]]) : (!FHE.eint<2>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V5]] : !FHE.ebool

  %1 = "FHE.gen_gate"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, tensor<4xi64>) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @and(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @and(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[TT:.*]] = arith.constant dense<[0, 0, 0, 1]> : tensor<4xi64>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 2 : i3
  // CHECK-NEXT: %[[V0:.*]] = "FHE.from_bool"(%arg0) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.from_bool"(%arg1) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%[[V0]], %[[C0]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint"(%[[V2]], %[[V1]]) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V4:.*]] = "FHE.apply_lookup_table"(%[[V3]], %[[TT]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V5:.*]] = "FHE.to_bool"(%[[V4]]) : (!FHE.eint<2>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V5]] : !FHE.ebool

  %1 = "FHE.and"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @or(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @or(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[TT:.*]] = arith.constant dense<[0, 1, 1, 1]> : tensor<4xi64>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 2 : i3
  // CHECK-NEXT: %[[V0:.*]] = "FHE.from_bool"(%arg0) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.from_bool"(%arg1) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%[[V0]], %[[C0]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint"(%[[V2]], %[[V1]]) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V4:.*]] = "FHE.apply_lookup_table"(%[[V3]], %[[TT]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V5:.*]] = "FHE.to_bool"(%[[V4]]) : (!FHE.eint<2>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V5]] : !FHE.ebool

  %1 = "FHE.or"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @nand(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @nand(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[TT:.*]] = arith.constant dense<[1, 1, 1, 0]> : tensor<4xi64>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 2 : i3
  // CHECK-NEXT: %[[V0:.*]] = "FHE.from_bool"(%arg0) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.from_bool"(%arg1) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%[[V0]], %[[C0]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint"(%[[V2]], %[[V1]]) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V4:.*]] = "FHE.apply_lookup_table"(%[[V3]], %[[TT]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V5:.*]] = "FHE.to_bool"(%[[V4]]) : (!FHE.eint<2>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V5]] : !FHE.ebool

  %1 = "FHE.nand"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @xor(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @xor(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[TT:.*]] = arith.constant dense<[0, 1, 1, 0]> : tensor<4xi64>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 2 : i3
  // CHECK-NEXT: %[[V0:.*]] = "FHE.from_bool"(%arg0) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.from_bool"(%arg1) : (!FHE.ebool) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%[[V0]], %[[C0]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V3:.*]] = "FHE.add_eint"(%[[V2]], %[[V1]]) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V4:.*]] = "FHE.apply_lookup_table"(%[[V3]], %[[TT]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: %[[V5:.*]] = "FHE.to_bool"(%[[V4]]) : (!FHE.eint<2>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V5]] : !FHE.ebool

  %1 = "FHE.xor"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}
