//===- SDFGOps.td - High level SDFG dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_SDFG_IR_SDFG_OPS
#define CONCRETELANG_DIALECT_SDFG_IR_SDFG_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "concretelang/Dialect/SDFG/IR/SDFGDialect.td"
include "concretelang/Dialect/SDFG/IR/SDFGTypes.td"

class SDFG_Op<string mnemonic, list<Trait> traits = []> :
    Op<SDFG_Dialect, mnemonic, traits>;

def StreamKindHostToDevice : I32EnumAttrCase<"host_to_device", 0>;
def StreamKindOnDevice : I32EnumAttrCase<"on_device", 1>;
def StreamKindDeviceToHost : I32EnumAttrCase<"device_to_host", 2>;
def StreamKindDeviceToBoth : I32EnumAttrCase<"device_to_both", 3>;

def StreamKind : I32EnumAttr<"StreamKind", "Stream kind",
  [StreamKindOnDevice, StreamKindHostToDevice, StreamKindDeviceToHost,
  StreamKindDeviceToBoth]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::concretelang::SDFG";
}

def StreamKindAttr : EnumAttr<SDFG_Dialect, StreamKind, "stream_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def SDFG_Init : SDFG_Op<"init"> {
    let summary = "Initializes the streaming framework";

    let description = [{
        Initializes the streaming framework. This operation must be
        performed before control reaches any other operation from the
        dialect.

        Example:
        ```mlir
        "SDFG.init" : () -> !SDFG.dfg
        ```
    }];

    let arguments = (ins);
    let results = (outs SDFG_DFG);
}

def SDFG_MakeStream : SDFG_Op<"make_stream"> {
    let summary = "Returns a new SDFG stream";

    let description = [{
        Returns a new SDFG stream, transporting data either between
        processes on the device, from the host to the device or from
        the device to the host. All streams are typed, allowing data
        to be read / written through `SDFG.get` and `SDFG.put` only
        using the stream's type.

        Example:
        ```mlir
        "SDFG.make_stream" { name = "stream", type = #SDFG.stream_kind<host_to_device> }(%dfg)
          : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
        ```
    }];

    let arguments = (ins SDFG_DFG:$dfg, StrAttr:$name, StreamKindAttr:$type);
    let results = (outs SDFG_Stream);

    let extraClassDeclaration = [{
      bool createsInputStream() {
        return getType() == StreamKind::host_to_device ||
          getType() == StreamKind::on_device ||
          getType() == StreamKind::device_to_both;
      }

      bool createsOutputStream() {
        return getType() == StreamKind::device_to_host ||
          getType() == StreamKind::on_device ||
          getType() == StreamKind::device_to_both;
      }
    }];
}

def ProcessKindAddEint : I32EnumAttrCase<"add_eint", 0>;
def ProcessKindAddEintInt : I32EnumAttrCase<"add_eint_int", 1>;
def ProcessKindMulEintInt : I32EnumAttrCase<"mul_eint_int", 2>;
def ProcessKindNegEint : I32EnumAttrCase<"neg_eint", 3>;
def ProcessKindKeyswitch : I32EnumAttrCase<"keyswitch", 4>;
def ProcessKindBootstrap : I32EnumAttrCase<"bootstrap", 5>;
def ProcessKindBatchAddEint : I32EnumAttrCase<"batched_add_eint", 6>;
def ProcessKindBatchAddEintInt : I32EnumAttrCase<"batched_add_eint_int", 7>;
def ProcessKindBatchAddEintIntCst : I32EnumAttrCase<"batched_add_eint_int_cst", 8>;
def ProcessKindBatchMulEintInt : I32EnumAttrCase<"batched_mul_eint_int", 9>;
def ProcessKindBatchMulEintIntCst : I32EnumAttrCase<"batched_mul_eint_int_cst", 10>;
def ProcessKindBatchNegEint : I32EnumAttrCase<"batched_neg_eint", 11>;
def ProcessKindBatchKeyswitch : I32EnumAttrCase<"batched_keyswitch", 12>;
def ProcessKindBatchBootstrap : I32EnumAttrCase<"batched_bootstrap", 13>;
def ProcessKindBatchMapBootstrap : I32EnumAttrCase<"batched_mapped_bootstrap", 14>;

def ProcessKind : I32EnumAttr<"ProcessKind", "Process kind",
  [ProcessKindAddEint, ProcessKindAddEintInt, ProcessKindMulEintInt,
   ProcessKindNegEint, ProcessKindKeyswitch, ProcessKindBootstrap,
   ProcessKindBatchAddEint, ProcessKindBatchAddEintInt,
   ProcessKindBatchAddEintIntCst, ProcessKindBatchMulEintInt,
   ProcessKindBatchMulEintIntCst, ProcessKindBatchNegEint,
   ProcessKindBatchKeyswitch, ProcessKindBatchBootstrap,
   ProcessKindBatchMapBootstrap]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::concretelang::SDFG";
}

def ProcessKindAttr : EnumAttr<SDFG_Dialect, ProcessKind, "process_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def SDFG_MakeProcess : SDFG_Op<"make_process"> {
    let summary = "Creates a new SDFG process";

    let description = [{
        Creates a new SDFG process and connects it to the input and
        output streams.

        Example:
        ```mlir
        %in0 = "SDFG.make_stream" { type = #SDFG.stream_kind<host_to_device> }(%dfg) : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
        %in1 = "SDFG.make_stream" { type = #SDFG.stream_kind<host_to_device> }(%dfg) : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
        %out = "SDFG.make_stream" { type = #SDFG.stream_kind<device_to_host> }(%dfg) : (!SDFG.dfg) -> !SDFG.stream<tensor<1024xi64>>
        "SDFG.make_process" { type = #SDFG.process_kind<add_eint> }(%dfg, %in0, %in1, %out) :
          (!SDFG.dfg, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>, !SDFG.stream<tensor<1024xi64>>) -> ()
        ```
    }];

    let extraClassDeclaration = [{
      ::mlir::LogicalResult checkStreams(size_t numIn, size_t numOut);
    }];

    let arguments = (ins ProcessKindAttr:$type, SDFG_DFG:$dfg, Variadic<SDFG_Stream>:$streams);
    let results = (outs);
    let hasVerifier = 1;
}

def SDFG_Put : SDFG_Op<"put"> {
    let summary = "Writes a data element to a stream";

    let description = [{
        Writes the input operand to the specified stream. The
        operand's type must meet the element type of the stream.

        Example:
        ```mlir
        "SDFG.put" (%stream, %data) : (!SDFG.stream<1024xi64>, tensor<1024xi64>) -> ()
        ```
    }];

    let arguments = (ins SDFG_Stream:$stream, AnyType:$data);
    let results = (outs);
    let hasVerifier = 1;
}

def SDFG_Get : SDFG_Op<"get"> {
    let summary = "Retrieves a data element from a stream";

    let description = [{
        Retrieves a single data element from the specified stream
        (i.e., an instance of the element type of the stream).

        Example:
        ```mlir
        "SDFG.get" (%stream) : (!SDFG.stream<1024xi64>) -> (tensor<1024xi64>)
        ```
    }];

    let arguments = (ins SDFG_Stream:$stream);
    let results = (outs AnyType:$data);
}

def SDFG_Start : SDFG_Op<"start"> {
    let summary = "Finalizes the creation of an SDFG and starts execution of its processes";


    let description = [{
        Finalizes the creation of an SDFG and starts execution of its
        processes. Any creation of streams and processes must take
        place before control reaches this operation.

        Example:
        ```mlir
        "SDFG.start"(%dfg) : !SDFG.dfg
        ```
    }];

    let arguments = (ins SDFG_DFG:$dfg);
    let results = (outs);
}


def SDFG_Shutdown : SDFG_Op<"shutdown"> {
    let summary = "Shuts down the streaming framework";

    let description = [{
        Shuts down the streaming framework. This operation must be
        performed after any other operation from the dialect.

        Example:
        ```mlir
        "SDFG.shutdown" (%dfg) : !SDFG.dfg
        ```
    }];

    let arguments = (ins SDFG_DFG:$dfg);
    let results = (outs);
}

#endif
