// RUN: concretecompiler --action=roundtrip --split-input-file %s 2>&1| FileCheck %s

// CHECK: func.func @init_shutdown
func.func @init_shutdown() -> () {
  // CHECK-NEXT: %[[DFG:.*]] = "SDFG.init"() : () -> !SDFG.dfg
  // CHECK-NEXT: "SDFG.shutdown"(%[[DFG]]) : (!SDFG.dfg) -> ()
  // CHECK-NEXT: return

  %dfg = "SDFG.init"() : () -> !SDFG.dfg
  "SDFG.shutdown"(%dfg) : (!SDFG.dfg) -> ()
  return
}

// -----

// CHECK:   func.func @simple_graph(%[[Varg0:.*]]: tensor<1024xi64>, %[[Varg1:.*]]: tensor<1024xi64>) -> tensor<1024xi64> {
// CHECK-NEXT:     %[[V0:.*]] = "SDFG.init"() : () -> !SDFG.dfg
// CHECK-NEXT:     %[[V1:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "in0", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
// CHECK-NEXT:     %[[V2:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "in1", type = #SDFG.stream_kind<host_to_device>} : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
// CHECK-NEXT:     %[[V3:.*]] = "SDFG.make_stream"(%[[V0]]) {name = "out", type = #SDFG.stream_kind<device_to_host>} : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
// CHECK-NEXT:     "SDFG.make_process"(%[[V0]], %[[V1]], %[[V2]], %[[V3]]) {type = #SDFG.process_kind<add_eint>} : (!SDFG.dfg, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>) -> ()
// CHECK-NEXT:     "SDFG.start"(%[[V0]]) : (!SDFG.dfg) -> ()
// CHECK-NEXT:     "SDFG.put"(%[[V1]], %[[Varg0]]) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
// CHECK-NEXT:     "SDFG.put"(%[[V2]], %[[Varg1]]) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
// CHECK-NEXT:     %[[V4:.*]] = "SDFG.get"(%[[V3]]) : (!SDFG.stream<tensor<1024xi64>>) -> tensor<1024xi64>
// CHECK-NEXT:     "SDFG.shutdown"(%[[V0]]) : (!SDFG.dfg) -> ()
// CHECK-NEXT:     return %[[V4]] : tensor<1024xi64>
// CHECK-NEXT:   }
func.func @simple_graph(%arg0: tensor<1024xi64>, %arg1: tensor<1024xi64>) -> tensor<1024xi64> {
  %dfg = "SDFG.init"() : () -> !SDFG.dfg
  %in0 = "SDFG.make_stream" (%dfg) { name = "in0", type = #SDFG.stream_kind<host_to_device> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  %in1 = "SDFG.make_stream" (%dfg) { name = "in1", type = #SDFG.stream_kind<host_to_device> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  %out = "SDFG.make_stream" (%dfg) { name = "out", type = #SDFG.stream_kind<device_to_host> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  "SDFG.make_process" (%dfg, %in0, %in1, %out) { type = #SDFG.process_kind<add_eint> } :
           (!SDFG.dfg, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>) -> ()
  "SDFG.start"(%dfg) : (!SDFG.dfg) -> ()

  "SDFG.put"(%in0, %arg0) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
  "SDFG.put"(%in1, %arg1) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
  %res = "SDFG.get"(%out) : (!SDFG.stream<tensor<1024xi64>>) -> tensor<1024xi64>

  "SDFG.shutdown"(%dfg) : (!SDFG.dfg) -> ()

   return %res : tensor<1024xi64>
}
