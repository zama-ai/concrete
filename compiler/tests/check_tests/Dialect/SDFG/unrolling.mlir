// RUN: concretecompiler --action=dump-sdfg --emit-sdfg-ops --unroll-loops-with-sdfg-convertible-ops --split-input-file %s 2>&1| FileCheck %s

// CHECK: func.func @main(%[[Varg0:.*]]: tensor<4x513xi64>, %[[Varg1:.*]]: tensor<4x513xi64>) -> tensor<4x513xi64> {
// CHECK-NEXT:   %[[V0:.*]] = "SDFG.init"() : () -> !SDFG.dfg
// CHECK-NEXT:   %[[V1:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream0", type = #SDFG.stream_kind<device_to_host>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V2:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream1", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V3:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream2", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   "SDFG.make_process"(%[[V0]], %[[V2]], %[[V3]], %[[V1]]) {type = #SDFG.process_kind<add_eint>} : (!SDFG.dfg, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>) -> ()
// CHECK-NEXT:   %[[V4:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream3", type = #SDFG.stream_kind<device_to_host>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V5:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream4", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V6:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream5", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   "SDFG.make_process"(%[[V0]], %[[V5]], %[[V6]], %[[V4]]) {type = #SDFG.process_kind<add_eint>} : (!SDFG.dfg, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>) -> ()
// CHECK-NEXT:   %[[V7:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream6", type = #SDFG.stream_kind<device_to_host>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V8:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream7", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V9:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream8", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   "SDFG.make_process"(%[[V0]], %[[V8]], %[[V9]], %[[V7]]) {type = #SDFG.process_kind<add_eint>} : (!SDFG.dfg, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>) -> ()
// CHECK-NEXT:   %[[V10:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream9", type = #SDFG.stream_kind<device_to_host>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V11:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream10", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   %[[V12:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "stream11", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<513xi64>>
// CHECK-NEXT:   "SDFG.make_process"(%[[V0]], %[[V11]], %[[V12]], %[[V10]]) {type = #SDFG.process_kind<add_eint>} : (!SDFG.dfg, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>, !SDFG.stream<tensor<513xi64>>) -> ()
// CHECK-NEXT:   "SDFG.start"(%[[V0]]) : (!SDFG.dfg) -> ()
// CHECK-NEXT:   %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[Vc1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[V13:.*]] = bufferization.alloc_tensor() : tensor<4x513xi64>
// CHECK-NEXT:   %[[V14:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[Vc0]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V15:.*]] = tensor.collapse_shape %[[V14]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V2]], %[[V15]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V16:.*]] = tensor.extract_slice %[[Varg1]]{{\[}}%[[Vc0]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V17:.*]] = tensor.collapse_shape %[[V16]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V3]], %[[V17]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V18:.*]] = "SDFG.get"(%[[V1]]) : (!SDFG.stream<tensor<513xi64>>) -> tensor<513xi64>
// CHECK-NEXT:   %[[V19:.*]] = tensor.insert_slice %[[V18]] into %[[V13]]{{\[}}%[[Vc0]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<513xi64> into tensor<4x513xi64>
// CHECK-NEXT:   %[[Vc1_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[V20:.*]] = arith.muli %[[Vc1]], %[[Vc1_0]] : index
// CHECK-NEXT:   %[[V21:.*]] = arith.addi %[[Vc0]], %[[V20]] : index
// CHECK-NEXT:   %[[V22:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[V21]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V23:.*]] = tensor.collapse_shape %[[V22]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V5]], %[[V23]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V24:.*]] = tensor.extract_slice %[[Varg1]]{{\[}}%[[V21]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V25:.*]] = tensor.collapse_shape %[[V24]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V6]], %[[V25]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V26:.*]] = "SDFG.get"(%[[V4]]) : (!SDFG.stream<tensor<513xi64>>) -> tensor<513xi64>
// CHECK-NEXT:   %[[V27:.*]] = tensor.insert_slice %[[V26]] into %[[V19]]{{\[}}%[[V21]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<513xi64> into tensor<4x513xi64>
// CHECK-NEXT:   %[[Vc2:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[V28:.*]] = arith.muli %[[Vc1]], %[[Vc2]] : index
// CHECK-NEXT:   %[[V29:.*]] = arith.addi %[[Vc0]], %[[V28]] : index
// CHECK-NEXT:   %[[V30:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[V29]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V31:.*]] = tensor.collapse_shape %[[V30]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V8]], %[[V31]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V32:.*]] = tensor.extract_slice %[[Varg1]]{{\[}}%[[V29]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V33:.*]] = tensor.collapse_shape %[[V32]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V9]], %[[V33]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V34:.*]] = "SDFG.get"(%[[V7]]) : (!SDFG.stream<tensor<513xi64>>) -> tensor<513xi64>
// CHECK-NEXT:   %[[V35:.*]] = tensor.insert_slice %[[V34]] into %[[V27]]{{\[}}%[[V29]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<513xi64> into tensor<4x513xi64>
// CHECK-NEXT:   %[[Vc3:.*]] = arith.constant 3 : index
// CHECK-NEXT:   %[[V36:.*]] = arith.muli %[[Vc1]], %[[Vc3]] : index
// CHECK-NEXT:   %[[V37:.*]] = arith.addi %[[Vc0]], %[[V36]] : index
// CHECK-NEXT:   %[[V38:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[V37]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V39:.*]] = tensor.collapse_shape %[[V38]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V11]], %[[V39]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V40:.*]] = tensor.extract_slice %[[Varg1]]{{\[}}%[[V37]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<4x513xi64> to tensor<1x513xi64>
// CHECK-NEXT:   %[[V41:.*]] = tensor.collapse_shape %[[V40]] {{\[\[0, 1\]\]}} : tensor<1x513xi64> into tensor<513xi64>
// CHECK-NEXT:   "SDFG.put"(%[[V12]], %[[V41]]) : (!SDFG.stream<tensor<513xi64>>, tensor<513xi64>) -> ()
// CHECK-NEXT:   %[[V42:.*]] = "SDFG.get"(%[[V10]]) : (!SDFG.stream<tensor<513xi64>>) -> tensor<513xi64>
// CHECK-NEXT:   %[[V43:.*]] = tensor.insert_slice %[[V42]] into %[[V35]]{{\[}}%[[V37]], 0{{\] \[1, 513\] \[1, 1\]}} : tensor<513xi64> into tensor<4x513xi64>
// CHECK-NEXT:   "SDFG.shutdown"(%[[V0]]) : (!SDFG.dfg) -> ()
// CHECK-NEXT:   return %[[V43]] : tensor<4x513xi64>
// CHECK-NEXT: }
func.func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4x!FHE.eint<6>>) -> tensor<4x!FHE.eint<6>> {
  %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4x!FHE.eint<6>>) -> tensor<4x!FHE.eint<6>>
  return %res : tensor<4x!FHE.eint<6>>
}
