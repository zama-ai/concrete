<!-- markdownlint-disable -->

# API Overview

## Modules

- [`concrete.compiler`](./concrete.compiler.md): Compiler submodule.
- [`concrete.compiler.compilation_context`](./concrete.compiler.compilation_context.md): CompilationContext.
- [`concrete.compiler.compilation_feedback`](./concrete.compiler.compilation_feedback.md): Compilation feedback.
- [`concrete.compiler.tfhers_int`](./concrete.compiler.tfhers_int.md): Import and export TFHErs integers into Concrete.
- [`concrete.compiler.utils`](./concrete.compiler.utils.md): Common utils for the compiler submodule.
- [`concrete.fhe`](./concrete.fhe.md): Concrete.
- [`concrete.fhe.compilation`](./concrete.fhe.compilation.md): Glue the compilation process together.
- [`concrete.fhe.compilation.artifacts`](./concrete.fhe.compilation.artifacts.md): Declaration of `DebugArtifacts` class.
- [`concrete.fhe.compilation.circuit`](./concrete.fhe.compilation.circuit.md): Declaration of `Circuit` class.
- [`concrete.fhe.compilation.client`](./concrete.fhe.compilation.client.md): Declaration of `Client` class.
- [`concrete.fhe.compilation.compiler`](./concrete.fhe.compilation.compiler.md): Declaration of `Compiler` class.
- [`concrete.fhe.compilation.composition`](./concrete.fhe.compilation.composition.md): Declaration of classes related to composition.
- [`concrete.fhe.compilation.configuration`](./concrete.fhe.compilation.configuration.md): Declaration of `Configuration` class.
- [`concrete.fhe.compilation.decorators`](./concrete.fhe.compilation.decorators.md): Declaration of `circuit` and `compiler` decorators.
- [`concrete.fhe.compilation.evaluation_keys`](./concrete.fhe.compilation.evaluation_keys.md): Declaration of `EvaluationKeys`.
- [`concrete.fhe.compilation.keys`](./concrete.fhe.compilation.keys.md): Declaration of `Keys` class.
- [`concrete.fhe.compilation.module`](./concrete.fhe.compilation.module.md): Declaration of `FheModule` classes.
- [`concrete.fhe.compilation.module_compiler`](./concrete.fhe.compilation.module_compiler.md): Declaration of `MultiCompiler` class.
- [`concrete.fhe.compilation.server`](./concrete.fhe.compilation.server.md): Declaration of `Server` class.
- [`concrete.fhe.compilation.specs`](./concrete.fhe.compilation.specs.md): Declaration of `ClientSpecs` class.
- [`concrete.fhe.compilation.status`](./concrete.fhe.compilation.status.md): Declaration of `EncryptionStatus` class.
- [`concrete.fhe.compilation.utils`](./concrete.fhe.compilation.utils.md): Declaration of various functions and constants related to compilation.
- [`concrete.fhe.compilation.value`](./concrete.fhe.compilation.value.md): Declaration of `Value` class.
- [`concrete.fhe.compilation.wiring`](./concrete.fhe.compilation.wiring.md): Declaration of wiring related class.
- [`concrete.fhe.dtypes`](./concrete.fhe.dtypes.md): Define available data types and their semantics.
- [`concrete.fhe.dtypes.base`](./concrete.fhe.dtypes.base.md): Declaration of `BaseDataType` abstract class.
- [`concrete.fhe.dtypes.float`](./concrete.fhe.dtypes.float.md): Declaration of `Float` class.
- [`concrete.fhe.dtypes.integer`](./concrete.fhe.dtypes.integer.md): Declaration of `Integer` class.
- [`concrete.fhe.dtypes.utils`](./concrete.fhe.dtypes.utils.md): Declaration of various functions and constants related to data types.
- [`concrete.fhe.extensions`](./concrete.fhe.extensions.md): Provide additional features that are not present in numpy.
- [`concrete.fhe.extensions.array`](./concrete.fhe.extensions.array.md): Declaration of `array` function, to simplify creation of encrypted arrays.
- [`concrete.fhe.extensions.bits`](./concrete.fhe.extensions.bits.md): Bit extraction extensions.
- [`concrete.fhe.extensions.constant`](./concrete.fhe.extensions.constant.md): Declaration of `constant` functions, to allow server side trivial encryption.
- [`concrete.fhe.extensions.convolution`](./concrete.fhe.extensions.convolution.md): Tracing and evaluation of convolution.
- [`concrete.fhe.extensions.hint`](./concrete.fhe.extensions.hint.md): Declaration of hinting extensions, to provide more information to Concrete.
- [`concrete.fhe.extensions.identity`](./concrete.fhe.extensions.identity.md): Declaration of `identity` extension.
- [`concrete.fhe.extensions.maxpool`](./concrete.fhe.extensions.maxpool.md): Tracing and evaluation of maxpool.
- [`concrete.fhe.extensions.multivariate`](./concrete.fhe.extensions.multivariate.md): Declaration of `multivariate` extension.
- [`concrete.fhe.extensions.ones`](./concrete.fhe.extensions.ones.md): Declaration of `ones` and `one` functions, to simplify creation of encrypted ones.
- [`concrete.fhe.extensions.relu`](./concrete.fhe.extensions.relu.md): Declaration of `relu` extension.
- [`concrete.fhe.extensions.round_bit_pattern`](./concrete.fhe.extensions.round_bit_pattern.md): Declaration of `round_bit_pattern` function, to provide an interface for rounded table lookups.
- [`concrete.fhe.extensions.table`](./concrete.fhe.extensions.table.md): Declaration of `LookupTable` class.
- [`concrete.fhe.extensions.tag`](./concrete.fhe.extensions.tag.md): Declaration of `tag` context manager, to allow tagging certain nodes.
- [`concrete.fhe.extensions.truncate_bit_pattern`](./concrete.fhe.extensions.truncate_bit_pattern.md): Declaration of `truncate_bit_pattern` extension.
- [`concrete.fhe.extensions.univariate`](./concrete.fhe.extensions.univariate.md): Declaration of `univariate` function.
- [`concrete.fhe.extensions.zeros`](./concrete.fhe.extensions.zeros.md): Declaration of `zeros` and `zero` functions, to simplify creation of encrypted zeros.
- [`concrete.fhe.internal`](./concrete.fhe.internal.md).
- [`concrete.fhe.internal.utils`](./concrete.fhe.internal.utils.md): Declaration of various functions and constants related to the entire project.
- [`concrete.fhe.mlir`](./concrete.fhe.mlir.md): Provide `computation graph` to `mlir` functionality.
- [`concrete.fhe.mlir.context`](./concrete.fhe.mlir.context.md): Declaration of `Context` class.
- [`concrete.fhe.mlir.conversion`](./concrete.fhe.mlir.conversion.md): Declaration of `ConversionType` and `Conversion` classes.
- [`concrete.fhe.mlir.converter`](./concrete.fhe.mlir.converter.md): Declaration of `Converter` class.
- [`concrete.fhe.mlir.processors`](./concrete.fhe.mlir.processors.md): All graph processors.
- [`concrete.fhe.mlir.processors.assign_bit_widths`](./concrete.fhe.mlir.processors.assign_bit_widths.md): Declaration of `AssignBitWidths` graph processor.
- [`concrete.fhe.mlir.processors.assign_node_ids`](./concrete.fhe.mlir.processors.assign_node_ids.md): Declaration of `AssignNodeIds` graph processor.
- [`concrete.fhe.mlir.processors.check_integer_only`](./concrete.fhe.mlir.processors.check_integer_only.md): Declaration of `CheckIntegerOnly` graph processor.
- [`concrete.fhe.mlir.processors.process_rounding`](./concrete.fhe.mlir.processors.process_rounding.md): Declaration of `ProcessRounding` graph processor.
- [`concrete.fhe.mlir.utils`](./concrete.fhe.mlir.utils.md): Declaration of various functions and constants related to MLIR conversion.
- [`concrete.fhe.representation`](./concrete.fhe.representation.md): Define structures used to represent computation.
- [`concrete.fhe.representation.evaluator`](./concrete.fhe.representation.evaluator.md): Declaration of various `Evaluator` classes, to make graphs picklable.
- [`concrete.fhe.representation.graph`](./concrete.fhe.representation.graph.md): Declaration of `Graph` class.
- [`concrete.fhe.representation.node`](./concrete.fhe.representation.node.md): Declaration of `Node` class.
- [`concrete.fhe.representation.operation`](./concrete.fhe.representation.operation.md): Declaration of `Operation` enum.
- [`concrete.fhe.representation.utils`](./concrete.fhe.representation.utils.md): Declaration of various functions and constants related to representation of computation.
- [`concrete.fhe.tfhers`](./concrete.fhe.tfhers.md): tfhers module to represent, and compute on tfhers integer values.
- [`concrete.fhe.tfhers.bridge`](./concrete.fhe.tfhers.bridge.md): Declaration of `tfhers.Bridge` class.
- [`concrete.fhe.tfhers.dtypes`](./concrete.fhe.tfhers.dtypes.md): Declaration of `TFHERSIntegerType` class.
- [`concrete.fhe.tfhers.specs`](./concrete.fhe.tfhers.specs.md): TFHE-rs client specs.
- [`concrete.fhe.tfhers.tracing`](./concrete.fhe.tfhers.tracing.md): Tracing of tfhers operations.
- [`concrete.fhe.tfhers.values`](./concrete.fhe.tfhers.values.md): Declaration of `TFHERSInteger` which wraps values as being of tfhers types.
- [`concrete.fhe.tracing`](./concrete.fhe.tracing.md): Provide `function` to `computation graph` functionality.
- [`concrete.fhe.tracing.tracer`](./concrete.fhe.tracing.tracer.md): Declaration of `Tracer` class.
- [`concrete.fhe.tracing.typing`](./concrete.fhe.tracing.typing.md): Declaration of type annotation.
- [`concrete.fhe.values`](./concrete.fhe.values.md): Define the available values and their semantics.
- [`concrete.fhe.values.scalar`](./concrete.fhe.values.scalar.md): Declaration of `ClearScalar` and `EncryptedScalar` wrappers.
- [`concrete.fhe.values.tensor`](./concrete.fhe.values.tensor.md): Declaration of `ClearTensor` and `EncryptedTensor` wrappers.
- [`concrete.fhe.values.value_description`](./concrete.fhe.values.value_description.md): Declaration of `ValueDescription` class.
- [`concrete.fhe.version`](./concrete.fhe.version.md): Version of the project, which is updated automatically by the CI right before releasing.
- [`concrete.lang`](./concrete.lang.md): Concretelang python module
- [`concrete.lang.dialects`](./concrete.lang.dialects.md)
- [`concrete.lang.dialects.fhe`](./concrete.lang.dialects.fhe.md): FHE dialect module
- [`concrete.lang.dialects.fhelinalg`](./concrete.lang.dialects.fhelinalg.md): FHELinalg dialect module
- [`concrete.lang.dialects.tracing`](./concrete.lang.dialects.tracing.md): Tracing dialect module

## Classes

- [`compiler.KeysetRestrictionHandler`](./concrete.compiler.md): Handler to serialize and deserialize keyset restrictions
- [`compiler.RangeRestrictionHandler`](./concrete.compiler.md): Handler to serialize and deserialize range restrictions
- [`compilation_context.CompilationContext`](./concrete.compiler.compilation_context.md): Compilation context.
- [`compilation_feedback.MoreCircuitCompilationFeedback`](./concrete.compiler.compilation_feedback.md): Helper class for compilation feedback.
- [`tfhers_int.TfhersExporter`](./concrete.compiler.tfhers_int.md): A helper class to import and export TFHErs big integers.
- [`artifacts.DebugArtifacts`](./concrete.fhe.compilation.artifacts.md): DebugArtifacts class, to export information about the compilation process for single function.
- [`artifacts.DebugManager`](./concrete.fhe.compilation.artifacts.md): A debug manager, allowing streamlined debugging.
- [`artifacts.FunctionDebugArtifacts`](./concrete.fhe.compilation.artifacts.md): An object containing debug artifacts for a certain function in an fhe module.
- [`artifacts.ModuleDebugArtifacts`](./concrete.fhe.compilation.artifacts.md): An object containing debug artifacts for an fhe module.
- [`circuit.Circuit`](./concrete.fhe.compilation.circuit.md): Circuit class, to combine computation graph, mlir, client and server into a single object.
- [`client.Client`](./concrete.fhe.compilation.client.md): Client class, which can be used to manage keys, encrypt arguments and decrypt results.
- [`compiler.Compiler`](./concrete.fhe.compilation.compiler.md): Compiler class, to glue the compilation pipeline.
- [`composition.CompositionClause`](./concrete.fhe.compilation.composition.md): A raw composition clause.
- [`composition.CompositionPolicy`](./concrete.fhe.compilation.composition.md): A protocol for composition policies.
- [`composition.CompositionRule`](./concrete.fhe.compilation.composition.md): A raw composition rule.
- [`configuration.ApproximateRoundingConfig`](./concrete.fhe.compilation.configuration.md): Controls the behavior of approximate rounding.
- [`configuration.BitwiseStrategy`](./concrete.fhe.compilation.configuration.md): BitwiseStrategy, to specify implementation preference for bitwise operations.
- [`configuration.ComparisonStrategy`](./concrete.fhe.compilation.configuration.md): ComparisonStrategy, to specify implementation preference for comparisons.
- [`configuration.Configuration`](./concrete.fhe.compilation.configuration.md): Configuration class, to allow the compilation process to be customized.
- [`configuration.Exactness`](./concrete.fhe.compilation.configuration.md).
- [`configuration.MinMaxStrategy`](./concrete.fhe.compilation.configuration.md): MinMaxStrategy, to specify implementation preference for minimum and maximum operations.
- [`configuration.MultiParameterStrategy`](./concrete.fhe.compilation.configuration.md): MultiParamStrategy, to set optimization strategy for multi-parameter.
- [`configuration.MultivariateStrategy`](./concrete.fhe.compilation.configuration.md): MultivariateStrategy, to specify implementation preference for multivariate operations.
- [`configuration.ParameterSelectionStrategy`](./concrete.fhe.compilation.configuration.md): ParameterSelectionStrategy, to set optimization strategy.
- [`configuration.SecurityLevel`](./concrete.fhe.compilation.configuration.md): Security level used to optimize the circuit parameters.
- [`decorators.Compilable`](./concrete.fhe.compilation.decorators.md): Compilable class, to wrap a function and provide methods to trace and compile it.
- [`evaluation_keys.EvaluationKeys`](./concrete.fhe.compilation.evaluation_keys.md): EvaluationKeys required for execution.
- [`keys.Keys`](./concrete.fhe.compilation.keys.md): Keys class, to manage generate/reuse keys.
- [`module.ExecutionRt`](./concrete.fhe.compilation.module.md): Runtime object class for execution.
- [`module.FheFunction`](./concrete.fhe.compilation.module.md): Fhe function class, allowing to run or simulate one function of an fhe module.
- [`module.FheModule`](./concrete.fhe.compilation.module.md): Fhe module class, to combine computation graphs, mlir, runtime objects into a single object.
- [`module.SimulationRt`](./concrete.fhe.compilation.module.md): Runtime object class for simulation.
- [`module_compiler.FunctionDef`](./concrete.fhe.compilation.module_compiler.md): An object representing the definition of a function as used in an fhe module.
- [`module_compiler.ModuleCompiler`](./concrete.fhe.compilation.module_compiler.md): Compiler class for multiple functions, to glue the compilation pipeline.
- [`server.Server`](./concrete.fhe.compilation.server.md): Server class, which can be used to perform homomorphic computation.
- [`specs.ClientSpecs`](./concrete.fhe.compilation.specs.md): ClientSpecs class, to create Client objects.
- [`status.EncryptionStatus`](./concrete.fhe.compilation.status.md): EncryptionStatus enum, to represent encryption status of parameters.
- [`utils.Lazy`](./concrete.fhe.compilation.utils.md): A lazyly initialized value.
- [`value.Value`](./concrete.fhe.compilation.value.md): A public value object that can be sent between client and server.
- [`wiring.AllComposable`](./concrete.fhe.compilation.wiring.md): Composition policy that allows to forward any output of the module to any of its input.
- [`wiring.AllInputs`](./concrete.fhe.compilation.wiring.md): All the encrypted inputs of a given function of a module.
- [`wiring.AllOutputs`](./concrete.fhe.compilation.wiring.md): All the encrypted outputs of a given function of a module.
- [`wiring.Input`](./concrete.fhe.compilation.wiring.md): The input of a given function of a module.
- [`wiring.NotComposable`](./concrete.fhe.compilation.wiring.md): Composition policy that does not allow the forwarding of any output to any input.
- [`wiring.Output`](./concrete.fhe.compilation.wiring.md): The output of a given function of a module.
- [`wiring.TracedOutput`](./concrete.fhe.compilation.wiring.md): A wrapper type used to trace wiring.
- [`wiring.Wire`](./concrete.fhe.compilation.wiring.md): A forwarding rule between an output and an input.
- [`wiring.WireInput`](./concrete.fhe.compilation.wiring.md): A protocol for wire inputs.
- [`wiring.WireOutput`](./concrete.fhe.compilation.wiring.md): A protocol for wire outputs.
- [`wiring.WireTracingContextManager`](./concrete.fhe.compilation.wiring.md): A context manager returned by the `wire_pipeline` method.
- [`wiring.Wired`](./concrete.fhe.compilation.wiring.md): Composition policy which allows the forwarding of certain outputs to certain inputs.
- [`base.BaseDataType`](./concrete.fhe.dtypes.base.md): BaseDataType abstract class, to form a basis for data types.
- [`float.Float`](./concrete.fhe.dtypes.float.md): Float class, to represent floating point numbers.
- [`integer.Integer`](./concrete.fhe.dtypes.integer.md): Integer class, to represent integers.
- [`bits.Bits`](./concrete.fhe.extensions.bits.md): Bits class, to provide indexing into the bits of integers.
- [`round_bit_pattern.Adjusting`](./concrete.fhe.extensions.round_bit_pattern.md): Adjusting class, to be used as early stop signal during adjustment.
- [`round_bit_pattern.AutoRounder`](./concrete.fhe.extensions.round_bit_pattern.md): AutoRounder class, to optimize for number of msbs to keep during round bit pattern operation.
- [`table.LookupTable`](./concrete.fhe.extensions.table.md): LookupTable class, to provide a way to do direct table lookups.
- [`truncate_bit_pattern.Adjusting`](./concrete.fhe.extensions.truncate_bit_pattern.md): Adjusting class, to be used as early stop signal during adjustment.
- [`truncate_bit_pattern.AutoTruncator`](./concrete.fhe.extensions.truncate_bit_pattern.md): AutoTruncator class, to optimize for the number of msbs to keep during truncate operation.
- [`context.Context`](./concrete.fhe.mlir.context.md): Context class, to perform operations on conversions.
- [`conversion.Conversion`](./concrete.fhe.mlir.conversion.md): Conversion class, to store MLIR operations with additional information.
- [`conversion.ConversionType`](./concrete.fhe.mlir.conversion.md): ConversionType class, to make it easier to work with MLIR types.
- [`converter.Converter`](./concrete.fhe.mlir.converter.md): Converter class, to convert a computation graph to MLIR.
- [`assign_bit_widths.AdditionalConstraints`](./concrete.fhe.mlir.processors.assign_bit_widths.md): AdditionalConstraints class to customize bit-width assignment step easily.
- [`assign_bit_widths.AssignBitWidths`](./concrete.fhe.mlir.processors.assign_bit_widths.md): AssignBitWidths graph processor, to assign proper bit-widths to be compatible with FHE.
- [`assign_node_ids.AssignNodeIds`](./concrete.fhe.mlir.processors.assign_node_ids.md) to node properties.
- [`check_integer_only.CheckIntegerOnly`](./concrete.fhe.mlir.processors.check_integer_only.md): CheckIntegerOnly graph processor, to make sure the graph only contains integer nodes.
- [`process_rounding.ProcessRounding`](./concrete.fhe.mlir.processors.process_rounding.md): ProcessRounding graph processor, to analyze rounding and support regular operations on it.
- [`utils.Comparison`](./concrete.fhe.mlir.utils.md): Comparison enum, to store the result comparison in 2-bits as there are three possible outcomes.
- [`utils.HashableNdarray`](./concrete.fhe.mlir.utils.md): HashableNdarray class, to use numpy arrays in dictionaries.
- [`evaluator.ConstantEvaluator`](./concrete.fhe.representation.evaluator.md): ConstantEvaluator class, to evaluate Operation.Constant nodes.
- [`evaluator.GenericEvaluator`](./concrete.fhe.representation.evaluator.md): GenericEvaluator class, to evaluate Operation.Generic nodes.
- [`evaluator.GenericTupleEvaluator`](./concrete.fhe.representation.evaluator.md): GenericEvaluator class, to evaluate Operation.Generic nodes where args are packed in a tuple.
- [`evaluator.InputEvaluator`](./concrete.fhe.representation.evaluator.md): InputEvaluator class, to evaluate Operation.Input nodes.
- [`graph.Graph`](./concrete.fhe.representation.graph.md): Graph class, to represent computation graphs.
- [`graph.GraphProcessor`](./concrete.fhe.representation.graph.md): GraphProcessor base class, to define the API for a graph processing pipeline.
- [`graph.MultiGraphProcessor`](./concrete.fhe.representation.graph.md): MultiGraphProcessor base class, to define the API for a multiple graph processing pipeline.
- [`node.Node`](./concrete.fhe.representation.node.md): Node class, to represent computation in a computation graph.
- [`operation.Operation`](./concrete.fhe.representation.operation.md): Operation enum, to distinguish nodes within a computation graph.
- [`bridge.Bridge`](./concrete.fhe.tfhers.bridge.md): TFHErs Bridge extend a Client with TFHErs functionalities.
- [`dtypes.CryptoParams`](./concrete.fhe.tfhers.dtypes.md): Crypto parameters used for a tfhers integer.
- [`dtypes.EncryptionKeyChoice`](./concrete.fhe.tfhers.dtypes.md): TFHErs key choice: big or small.
- [`dtypes.TFHERSIntegerType`](./concrete.fhe.tfhers.dtypes.md) to represent tfhers integer types.
- [`specs.TFHERSClientSpecs`](./concrete.fhe.tfhers.specs.md): TFHE-rs client specs.
- [`values.TFHERSInteger`](./concrete.fhe.tfhers.values.md) into typed values, using tfhers types.
- [`tracer.Annotation`](./concrete.fhe.tracing.tracer.md): Base annotation for direct definition.
- [`tracer.ScalarAnnotation`](./concrete.fhe.tracing.tracer.md): Base scalar annotation for direct definition.
- [`tracer.TensorAnnotation`](./concrete.fhe.tracing.tracer.md): Base tensor annotation for direct definition.
- [`tracer.Tracer`](./concrete.fhe.tracing.tracer.md): Tracer class, to create computation graphs from python functions.
- [`typing.f32`](./concrete.fhe.tracing.typing.md): Scalar f32 annotation.
- [`typing.f64`](./concrete.fhe.tracing.typing.md): Scalar f64 annotation.
- [`typing.int1`](./concrete.fhe.tracing.typing.md): Scalar int1 annotation.
- [`typing.int10`](./concrete.fhe.tracing.typing.md): Scalar int10 annotation.
- [`typing.int11`](./concrete.fhe.tracing.typing.md): Scalar int11 annotation.
- [`typing.int12`](./concrete.fhe.tracing.typing.md): Scalar int12 annotation.
- [`typing.int13`](./concrete.fhe.tracing.typing.md): Scalar int13 annotation.
- [`typing.int14`](./concrete.fhe.tracing.typing.md): Scalar int14 annotation.
- [`typing.int15`](./concrete.fhe.tracing.typing.md): Scalar int15 annotation.
- [`typing.int16`](./concrete.fhe.tracing.typing.md): Scalar int16 annotation.
- [`typing.int17`](./concrete.fhe.tracing.typing.md): Scalar int17 annotation.
- [`typing.int18`](./concrete.fhe.tracing.typing.md): Scalar int18 annotation.
- [`typing.int19`](./concrete.fhe.tracing.typing.md): Scalar int19 annotation.
- [`typing.int2`](./concrete.fhe.tracing.typing.md): Scalar int2 annotation.
- [`typing.int20`](./concrete.fhe.tracing.typing.md): Scalar int20 annotation.
- [`typing.int21`](./concrete.fhe.tracing.typing.md): Scalar int21 annotation.
- [`typing.int22`](./concrete.fhe.tracing.typing.md): Scalar int22 annotation.
- [`typing.int23`](./concrete.fhe.tracing.typing.md): Scalar int23 annotation.
- [`typing.int24`](./concrete.fhe.tracing.typing.md): Scalar int24 annotation.
- [`typing.int25`](./concrete.fhe.tracing.typing.md): Scalar int25 annotation.
- [`typing.int26`](./concrete.fhe.tracing.typing.md): Scalar int26 annotation.
- [`typing.int27`](./concrete.fhe.tracing.typing.md): Scalar int27 annotation.
- [`typing.int28`](./concrete.fhe.tracing.typing.md): Scalar int28 annotation.
- [`typing.int29`](./concrete.fhe.tracing.typing.md): Scalar int29 annotation.
- [`typing.int3`](./concrete.fhe.tracing.typing.md): Scalar int3 annotation.
- [`typing.int30`](./concrete.fhe.tracing.typing.md): Scalar int30 annotation.
- [`typing.int31`](./concrete.fhe.tracing.typing.md): Scalar int31 annotation.
- [`typing.int32`](./concrete.fhe.tracing.typing.md): Scalar int32 annotation.
- [`typing.int33`](./concrete.fhe.tracing.typing.md): Scalar int33 annotation.
- [`typing.int34`](./concrete.fhe.tracing.typing.md): Scalar int34 annotation.
- [`typing.int35`](./concrete.fhe.tracing.typing.md): Scalar int35 annotation.
- [`typing.int36`](./concrete.fhe.tracing.typing.md): Scalar int36 annotation.
- [`typing.int37`](./concrete.fhe.tracing.typing.md): Scalar int37 annotation.
- [`typing.int38`](./concrete.fhe.tracing.typing.md): Scalar int38 annotation.
- [`typing.int39`](./concrete.fhe.tracing.typing.md): Scalar int39 annotation.
- [`typing.int4`](./concrete.fhe.tracing.typing.md): Scalar int4 annotation.
- [`typing.int40`](./concrete.fhe.tracing.typing.md): Scalar int40 annotation.
- [`typing.int41`](./concrete.fhe.tracing.typing.md): Scalar int41 annotation.
- [`typing.int42`](./concrete.fhe.tracing.typing.md): Scalar int42 annotation.
- [`typing.int43`](./concrete.fhe.tracing.typing.md): Scalar int43 annotation.
- [`typing.int44`](./concrete.fhe.tracing.typing.md): Scalar int44 annotation.
- [`typing.int45`](./concrete.fhe.tracing.typing.md): Scalar int45 annotation.
- [`typing.int46`](./concrete.fhe.tracing.typing.md): Scalar int46 annotation.
- [`typing.int47`](./concrete.fhe.tracing.typing.md): Scalar int47 annotation.
- [`typing.int48`](./concrete.fhe.tracing.typing.md): Scalar int48 annotation.
- [`typing.int49`](./concrete.fhe.tracing.typing.md): Scalar int49 annotation.
- [`typing.int5`](./concrete.fhe.tracing.typing.md): Scalar int5 annotation.
- [`typing.int50`](./concrete.fhe.tracing.typing.md): Scalar int50 annotation.
- [`typing.int51`](./concrete.fhe.tracing.typing.md): Scalar int51 annotation.
- [`typing.int52`](./concrete.fhe.tracing.typing.md): Scalar int52 annotation.
- [`typing.int53`](./concrete.fhe.tracing.typing.md): Scalar int53 annotation.
- [`typing.int54`](./concrete.fhe.tracing.typing.md): Scalar int54 annotation.
- [`typing.int55`](./concrete.fhe.tracing.typing.md): Scalar int55 annotation.
- [`typing.int56`](./concrete.fhe.tracing.typing.md): Scalar int56 annotation.
- [`typing.int57`](./concrete.fhe.tracing.typing.md): Scalar int57 annotation.
- [`typing.int58`](./concrete.fhe.tracing.typing.md): Scalar int58 annotation.
- [`typing.int59`](./concrete.fhe.tracing.typing.md): Scalar int59 annotation.
- [`typing.int6`](./concrete.fhe.tracing.typing.md): Scalar int6 annotation.
- [`typing.int60`](./concrete.fhe.tracing.typing.md): Scalar int60 annotation.
- [`typing.int61`](./concrete.fhe.tracing.typing.md): Scalar int61 annotation.
- [`typing.int62`](./concrete.fhe.tracing.typing.md): Scalar int62 annotation.
- [`typing.int63`](./concrete.fhe.tracing.typing.md): Scalar int63 annotation.
- [`typing.int64`](./concrete.fhe.tracing.typing.md): Scalar int64 annotation.
- [`typing.int7`](./concrete.fhe.tracing.typing.md): Scalar int7 annotation.
- [`typing.int8`](./concrete.fhe.tracing.typing.md): Scalar int8 annotation.
- [`typing.int9`](./concrete.fhe.tracing.typing.md): Scalar int9 annotation.
- [`typing.tensor`](./concrete.fhe.tracing.typing.md): Tensor annotation.
- [`typing.uint1`](./concrete.fhe.tracing.typing.md): Scalar uint1 annotation.
- [`typing.uint10`](./concrete.fhe.tracing.typing.md): Scalar uint10 annotation.
- [`typing.uint11`](./concrete.fhe.tracing.typing.md): Scalar uint11 annotation.
- [`typing.uint12`](./concrete.fhe.tracing.typing.md): Scalar uint12 annotation.
- [`typing.uint13`](./concrete.fhe.tracing.typing.md): Scalar uint13 annotation.
- [`typing.uint14`](./concrete.fhe.tracing.typing.md): Scalar uint14 annotation.
- [`typing.uint15`](./concrete.fhe.tracing.typing.md): Scalar uint15 annotation.
- [`typing.uint16`](./concrete.fhe.tracing.typing.md): Scalar uint16 annotation.
- [`typing.uint17`](./concrete.fhe.tracing.typing.md): Scalar uint17 annotation.
- [`typing.uint18`](./concrete.fhe.tracing.typing.md): Scalar uint18 annotation.
- [`typing.uint19`](./concrete.fhe.tracing.typing.md): Scalar uint19 annotation.
- [`typing.uint2`](./concrete.fhe.tracing.typing.md): Scalar uint2 annotation.
- [`typing.uint20`](./concrete.fhe.tracing.typing.md): Scalar uint20 annotation.
- [`typing.uint21`](./concrete.fhe.tracing.typing.md): Scalar uint21 annotation.
- [`typing.uint22`](./concrete.fhe.tracing.typing.md): Scalar uint22 annotation.
- [`typing.uint23`](./concrete.fhe.tracing.typing.md): Scalar uint23 annotation.
- [`typing.uint24`](./concrete.fhe.tracing.typing.md): Scalar uint24 annotation.
- [`typing.uint25`](./concrete.fhe.tracing.typing.md): Scalar uint25 annotation.
- [`typing.uint26`](./concrete.fhe.tracing.typing.md): Scalar uint26 annotation.
- [`typing.uint27`](./concrete.fhe.tracing.typing.md): Scalar uint27 annotation.
- [`typing.uint28`](./concrete.fhe.tracing.typing.md): Scalar uint28 annotation.
- [`typing.uint29`](./concrete.fhe.tracing.typing.md): Scalar uint29 annotation.
- [`typing.uint3`](./concrete.fhe.tracing.typing.md): Scalar uint3 annotation.
- [`typing.uint30`](./concrete.fhe.tracing.typing.md): Scalar uint30 annotation.
- [`typing.uint31`](./concrete.fhe.tracing.typing.md): Scalar uint31 annotation.
- [`typing.uint32`](./concrete.fhe.tracing.typing.md): Scalar uint32 annotation.
- [`typing.uint33`](./concrete.fhe.tracing.typing.md): Scalar uint33 annotation.
- [`typing.uint34`](./concrete.fhe.tracing.typing.md): Scalar uint34 annotation.
- [`typing.uint35`](./concrete.fhe.tracing.typing.md): Scalar uint35 annotation.
- [`typing.uint36`](./concrete.fhe.tracing.typing.md): Scalar uint36 annotation.
- [`typing.uint37`](./concrete.fhe.tracing.typing.md): Scalar uint37 annotation.
- [`typing.uint38`](./concrete.fhe.tracing.typing.md): Scalar uint38 annotation.
- [`typing.uint39`](./concrete.fhe.tracing.typing.md): Scalar uint39 annotation.
- [`typing.uint4`](./concrete.fhe.tracing.typing.md): Scalar uint4 annotation.
- [`typing.uint40`](./concrete.fhe.tracing.typing.md): Scalar uint40 annotation.
- [`typing.uint41`](./concrete.fhe.tracing.typing.md): Scalar uint41 annotation.
- [`typing.uint42`](./concrete.fhe.tracing.typing.md): Scalar uint42 annotation.
- [`typing.uint43`](./concrete.fhe.tracing.typing.md): Scalar uint43 annotation.
- [`typing.uint44`](./concrete.fhe.tracing.typing.md): Scalar uint44 annotation.
- [`typing.uint45`](./concrete.fhe.tracing.typing.md): Scalar uint45 annotation.
- [`typing.uint46`](./concrete.fhe.tracing.typing.md): Scalar uint46 annotation.
- [`typing.uint47`](./concrete.fhe.tracing.typing.md): Scalar uint47 annotation.
- [`typing.uint48`](./concrete.fhe.tracing.typing.md): Scalar uint48 annotation.
- [`typing.uint49`](./concrete.fhe.tracing.typing.md): Scalar uint49 annotation.
- [`typing.uint5`](./concrete.fhe.tracing.typing.md): Scalar uint5 annotation.
- [`typing.uint50`](./concrete.fhe.tracing.typing.md): Scalar uint50 annotation.
- [`typing.uint51`](./concrete.fhe.tracing.typing.md): Scalar uint51 annotation.
- [`typing.uint52`](./concrete.fhe.tracing.typing.md): Scalar uint52 annotation.
- [`typing.uint53`](./concrete.fhe.tracing.typing.md): Scalar uint53 annotation.
- [`typing.uint54`](./concrete.fhe.tracing.typing.md): Scalar uint54 annotation.
- [`typing.uint55`](./concrete.fhe.tracing.typing.md): Scalar uint55 annotation.
- [`typing.uint56`](./concrete.fhe.tracing.typing.md): Scalar uint56 annotation.
- [`typing.uint57`](./concrete.fhe.tracing.typing.md): Scalar uint57 annotation.
- [`typing.uint58`](./concrete.fhe.tracing.typing.md): Scalar uint58 annotation.
- [`typing.uint59`](./concrete.fhe.tracing.typing.md): Scalar uint59 annotation.
- [`typing.uint6`](./concrete.fhe.tracing.typing.md): Scalar uint6 annotation.
- [`typing.uint60`](./concrete.fhe.tracing.typing.md): Scalar uint60 annotation.
- [`typing.uint61`](./concrete.fhe.tracing.typing.md): Scalar uint61 annotation.
- [`typing.uint62`](./concrete.fhe.tracing.typing.md): Scalar uint62 annotation.
- [`typing.uint63`](./concrete.fhe.tracing.typing.md): Scalar uint63 annotation.
- [`typing.uint64`](./concrete.fhe.tracing.typing.md): Scalar uint64 annotation.
- [`typing.uint7`](./concrete.fhe.tracing.typing.md): Scalar uint7 annotation.
- [`typing.uint8`](./concrete.fhe.tracing.typing.md): Scalar uint8 annotation.
- [`typing.uint9`](./concrete.fhe.tracing.typing.md): Scalar uint9 annotation.
- [`value_description.ValueDescription`](./concrete.fhe.values.value_description.md): ValueDescription class, to combine data type, shape, and encryption status into a single object.

## Functions

- [`compiler.check_gpu_available`](./concrete.compiler.md): Check whether a CUDA device is available and online.
- [`compiler.check_gpu_enabled`](./concrete.compiler.md): Check whether the compiler and runtime support GPU offloading.
- [`compiler.init_dfr`](./concrete.compiler.md): Initialize dataflow parallelization.
- [`compiler.round_trip`](./concrete.compiler.md): Parse the MLIR input, then return it back.
- [`compilation_feedback.tag_from_location`](./concrete.compiler.compilation_feedback.md): Extract tag of the operation from its location.
- [`utils.lookup_runtime_lib`](./concrete.compiler.utils.md): Try to find the absolute path to the runtime library.
- [`decorators.circuit`](./concrete.fhe.compilation.decorators.md): Provide a direct interface for compilation of single circuit programs.
- [`decorators.compiler`](./concrete.fhe.compilation.decorators.md): Provide an easy interface for the compilation of single-circuit programs.
- [`decorators.function`](./concrete.fhe.compilation.decorators.md): Provide an easy interface to define a function within an fhe module.
- [`decorators.module`](./concrete.fhe.compilation.decorators.md): Provide an easy interface for the compilation of multi functions modules.
- [`utils.add_nodes_from_to`](./concrete.fhe.compilation.utils.md): Add nodes from `from_nodes` to `to_nodes`, to `all_nodes`.
- [`utils.check_subgraph_fusibility`](./concrete.fhe.compilation.utils.md): Determine if a subgraph can be fused.
- [`utils.convert_subgraph_to_subgraph_node`](./concrete.fhe.compilation.utils.md): Convert a subgraph to Operation.Generic node.
- [`utils.find_closest_integer_output_nodes`](./concrete.fhe.compilation.utils.md): Find the closest upstream integer output nodes to a set of start nodes in a graph.
- [`utils.find_float_subgraph_with_unique_terminal_node`](./concrete.fhe.compilation.utils.md): Find a subgraph with float computations that end with an integer output.
- [`utils.find_single_lca`](./concrete.fhe.compilation.utils.md): Find the single lowest common ancestor of a list of nodes.
- [`utils.find_tlu_subgraph_with_multiple_variable_inputs_that_has_a_single_common_ancestor`](./concrete.fhe.compilation.utils.md): Find a subgraph with a tlu computation that has multiple variable inputs     where all variable inputs share a common ancestor.
- [`utils.friendly_type_format`](./concrete.fhe.compilation.utils.md): Convert a type to a string. Remove package name and class/type keywords.
- [`utils.fuse`](./concrete.fhe.compilation.utils.md): Fuse appropriate subgraphs in a graph to a single Operation.Generic node.
- [`utils.get_terminal_size`](./concrete.fhe.compilation.utils.md): Get the terminal size.
- [`utils.inputset`](./concrete.fhe.compilation.utils.md): Generate a random inputset.
- [`utils.is_single_common_ancestor`](./concrete.fhe.compilation.utils.md): Determine if a node is the single common ancestor of a list of nodes.
- [`utils.validate_input_args`](./concrete.fhe.compilation.utils.md): Validate input arguments.
- [`utils.combine_dtypes`](./concrete.fhe.dtypes.utils.md): Get the 'BaseDataType' that can represent a set of 'BaseDataType's.
- [`array.array`](./concrete.fhe.extensions.array.md): Create an encrypted array from either encrypted or clear values.
- [`bits.bits`](./concrete.fhe.extensions.bits.md): Extract bits of integers.
- [`constant.constant`](./concrete.fhe.extensions.constant.md): Trivial encryption of a cleartext value.
- [`convolution.conv`](./concrete.fhe.extensions.convolution.md): Trace and evaluate convolution operations.
- [`hint.hint`](./concrete.fhe.extensions.hint.md): Hint the compilation process about properties of a value.
- [`identity.identity`](./concrete.fhe.extensions.identity.md): Apply identity function to x.
- [`identity.refresh`](./concrete.fhe.extensions.identity.md): Refresh x.
- [`maxpool.maxpool`](./concrete.fhe.extensions.maxpool.md): Evaluate or trace MaxPool operation.
- [`multivariate.multivariate`](./concrete.fhe.extensions.multivariate.md): Wrap a multivariate function so that it is traced into a single generic node.
- [`ones.one`](./concrete.fhe.extensions.ones.md): Create an encrypted scalar with the value of one.
- [`ones.ones`](./concrete.fhe.extensions.ones.md): Create an encrypted array of ones.
- [`ones.ones_like`](./concrete.fhe.extensions.ones.md): Create an encrypted array of ones with the same shape as another array.
- [`relu.relu`](./concrete.fhe.extensions.relu.md): Rectified linear unit extension.
- [`round_bit_pattern.round_bit_pattern`](./concrete.fhe.extensions.round_bit_pattern.md): Round the bit pattern of an integer.
- [`tag.tag`](./concrete.fhe.extensions.tag.md): Introduce a new tag to the tag stack.
- [`truncate_bit_pattern.truncate_bit_pattern`](./concrete.fhe.extensions.truncate_bit_pattern.md): Round the bit pattern of an integer.
- [`univariate.univariate`](./concrete.fhe.extensions.univariate.md): Wrap a univariate function so that it is traced into a single generic node.
- [`zeros.zero`](./concrete.fhe.extensions.zeros.md): Create an encrypted scalar with the value of zero.
- [`zeros.zeros`](./concrete.fhe.extensions.zeros.md): Create an encrypted array of zeros.
- [`zeros.zeros_like`](./concrete.fhe.extensions.zeros.md): Create an encrypted array of zeros with the same shape as another array.
- [`utils.assert_that`](./concrete.fhe.internal.utils.md): Assert a condition.
- [`utils.unreachable`](./concrete.fhe.internal.utils.md): Raise a RuntimeError to indicate unreachable code is entered.
- [`utils.construct_deduplicated_tables`](./concrete.fhe.mlir.utils.md): Construct lookup tables for each cell of the input for an Operation.Generic node.
- [`utils.construct_table`](./concrete.fhe.mlir.utils.md): Construct the lookup table for an Operation.Generic node.
- [`utils.construct_table_multivariate`](./concrete.fhe.mlir.utils.md): Construct the lookup table for a multivariate node.
- [`utils.flood_replace_none_values`](./concrete.fhe.mlir.utils.md): Use flooding algorithm to replace `None` values.
- [`utils.format_constant`](./concrete.fhe.representation.utils.md): Get the textual representation of a constant.
- [`utils.format_indexing_element`](./concrete.fhe.representation.utils.md): Format an indexing element.
- [`tfhers.get_type_from_params`](./concrete.fhe.tfhers.md): Get a TFHE-rs integer type from TFHE-rs parameters in JSON format.
- [`tfhers.get_type_from_params_dict`](./concrete.fhe.tfhers.md): Get a TFHE-rs integer type from TFHE-rs parameters in JSON format.
- [`bridge.new_bridge`](./concrete.fhe.tfhers.bridge.md): Create a TFHErs bridge from a circuit or module or client.
- [`tracing.from_native`](./concrete.fhe.tfhers.tracing.md): Convert a Concrete integer to the tfhers representation.
- [`tracing.to_native`](./concrete.fhe.tfhers.tracing.md): Convert a tfhers integer to the Concrete representation.
- [`scalar.clear_scalar_builder`](./concrete.fhe.values.scalar.md): Build a clear scalar value.
- [`scalar.encrypted_scalar_builder`](./concrete.fhe.values.scalar.md): Build an encrypted scalar value.
- [`scalar.clear_scalar_builder`](./concrete.fhe.values.scalar.md): Build a clear scalar value.
- [`scalar.encrypted_scalar_builder`](./concrete.fhe.values.scalar.md): Build an encrypted scalar value.
- [`tensor.clear_tensor_builder`](./concrete.fhe.values.tensor.md): Build a clear tensor value.
- [`tensor.encrypted_tensor_builder`](./concrete.fhe.values.tensor.md): Build an encrypted tensor value.
- [`tensor.clear_tensor_builder`](./concrete.fhe.values.tensor.md): Build a clear tensor value.
- [`tensor.encrypted_tensor_builder`](./concrete.fhe.values.tensor.md): Build an encrypted tensor value.
