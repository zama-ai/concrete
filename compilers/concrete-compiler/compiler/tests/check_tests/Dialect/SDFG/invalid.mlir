// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

func.func @wrong_element_type(%arg0: tensor<2xi32>, %arg1: tensor<1024xi64>) -> tensor<1024xi64> {
  %dfg = "SDFG.init"() : () -> !SDFG.dfg
  %in0 = "SDFG.make_stream" (%dfg) { name = "in0", type = #SDFG.stream_kind<host_to_device> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  %in1 = "SDFG.make_stream" (%dfg) { name = "in1", type = #SDFG.stream_kind<host_to_device> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  %out = "SDFG.make_stream" (%dfg) { name = "out", type = #SDFG.stream_kind<device_to_host> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  "SDFG.make_process" (%dfg, %in0, %in1, %out) { type = #SDFG.process_kind<add_eint> } :
           (!SDFG.dfg, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>) -> ()
  "SDFG.start"(%dfg) : (!SDFG.dfg) -> ()

  // expected-error @+1 {{The type 'tensor<2xi32>' of the element to be written does not match the element type 'tensor<1024xi64>' of the stream.}}
  "SDFG.put"(%in0, %arg0) : (!SDFG.stream<tensor<1024xi64>>, tensor<2xi32>) -> ()
  "SDFG.put"(%in1, %arg1) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
  %res = "SDFG.get"(%out) : (!SDFG.stream<tensor<1024xi64>>) -> tensor<1024xi64>

  "SDFG.shutdown"(%dfg) : (!SDFG.dfg) -> ()

   return %res : tensor<1024xi64>
}

// -----

func.func @wrong_stream_direction(%arg0: tensor<1024xi64>, %arg1: tensor<1024xi64>) -> tensor<1024xi64> {
  %dfg = "SDFG.init"() : () -> !SDFG.dfg
  %in0 = "SDFG.make_stream" (%dfg) { name = "inXXX0", type = #SDFG.stream_kind<device_to_host> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  %in1 = "SDFG.make_stream" (%dfg) { name = "in1", type = #SDFG.stream_kind<host_to_device> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  %out = "SDFG.make_stream" (%dfg) { name = "out", type = #SDFG.stream_kind<device_to_host> } : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
  // expected-error @+1 {{Stream #1 of process `add_eint` must be an input stream.}}
  "SDFG.make_process" (%dfg, %in0, %in1, %out) { type = #SDFG.process_kind<add_eint> } :
           (!SDFG.dfg, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>) -> ()
  "SDFG.start"(%dfg) : (!SDFG.dfg) -> ()

  "SDFG.put"(%in0, %arg0) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
  "SDFG.put"(%in1, %arg1) : (!SDFG.stream<tensor<1024xi64>>, tensor<1024xi64>) -> ()
  %res = "SDFG.get"(%out) : (!SDFG.stream<tensor<1024xi64>>) -> tensor<1024xi64>

  "SDFG.shutdown"(%dfg) : (!SDFG.dfg) -> ()

   return %res : tensor<1024xi64>
}
