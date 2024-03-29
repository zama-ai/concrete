<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'SDFG' Dialect

Dialect for the construction of static data flow graphs
A dialect for the construction of static data flow graphs. The
data flow graph is composed of a set of processes, connected
through data streams. Special streams allow for data to be
injected into and to be retrieved from the data flow graph.



## Operation definition

### `SDFG.get` (::mlir::concretelang::SDFG::Get)

Retrieves a data element from a stream

Retrieves a single data element from the specified stream
(i.e., an instance of the element type of the stream).

Example:
```mlir
"SDFG.get" (%stream) : (!SDFG.stream<1024xi64>) -> (tensor<1024xi64>)
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `stream` | An SDFG data stream

#### Results:

| Result | Description |
| :----: | ----------- |
| `data` | any type

### `SDFG.init` (::mlir::concretelang::SDFG::Init)

Initializes the streaming framework

Initializes the streaming framework. This operation must be
performed before control reaches any other operation from the
dialect.

Example:
```mlir
"SDFG.init" : () -> !SDFG.dfg
```

#### Results:

| Result | Description |
| :----: | ----------- |
&laquo;unnamed&raquo; | An SDFG data flow graph

### `SDFG.make_process` (::mlir::concretelang::SDFG::MakeProcess)

Creates a new SDFG process

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

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `type` | ::mlir::concretelang::SDFG::ProcessKindAttr | Process kind

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dfg` | An SDFG data flow graph
| `streams` | An SDFG data stream

### `SDFG.make_stream` (::mlir::concretelang::SDFG::MakeStream)

Returns a new SDFG stream

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

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `name` | ::mlir::StringAttr | string attribute
| `type` | ::mlir::concretelang::SDFG::StreamKindAttr | Stream kind

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dfg` | An SDFG data flow graph

#### Results:

| Result | Description |
| :----: | ----------- |
&laquo;unnamed&raquo; | An SDFG data stream

### `SDFG.put` (::mlir::concretelang::SDFG::Put)

Writes a data element to a stream

Writes the input operand to the specified stream. The
operand's type must meet the element type of the stream.

Example:
```mlir
"SDFG.put" (%stream, %data) : (!SDFG.stream<1024xi64>, tensor<1024xi64>) -> ()
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `stream` | An SDFG data stream
| `data` | any type

### `SDFG.shutdown` (::mlir::concretelang::SDFG::Shutdown)

Shuts down the streaming framework

Shuts down the streaming framework. This operation must be
performed after any other operation from the dialect.

Example:
```mlir
"SDFG.shutdown" (%dfg) : !SDFG.dfg
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dfg` | An SDFG data flow graph

### `SDFG.start` (::mlir::concretelang::SDFG::Start)

Finalizes the creation of an SDFG and starts execution of its processes

Finalizes the creation of an SDFG and starts execution of its
processes. Any creation of streams and processes must take
place before control reaches this operation.

Example:
```mlir
"SDFG.start"(%dfg) : !SDFG.dfg
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dfg` | An SDFG data flow graph

## Attribute definition

### ProcessKindAttr

Process kind

Syntax:

```
#SDFG.process_kind<
  ::mlir::concretelang::SDFG::ProcessKind   # value
>
```


#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::concretelang::SDFG::ProcessKind` | an enum of type ProcessKind |

### StreamKindAttr

Stream kind

Syntax:

```
#SDFG.stream_kind<
  ::mlir::concretelang::SDFG::StreamKind   # value
>
```


#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| value | `::mlir::concretelang::SDFG::StreamKind` | an enum of type StreamKind |

## Type definition

### DFGType

An SDFG data flow graph

Syntax: `!SDFG.dfg`

A handle to an SDFG data flow graph

### StreamType

An SDFG data stream

An SDFG stream to connect SDFG processes.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| elementType | `Type` |  |

