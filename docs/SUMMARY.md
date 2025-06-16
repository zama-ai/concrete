# Table of contents

* [Welcome](README.md)

## Get Started

* [What is Concrete?](get-started/readme.md)
* [Installation](get-started/installing.md)
* [Quick start](get-started/quick_start.md)
* [Quick overview](get-started/quick_overview.md)
* [Terminology](get-started/terminology.md)

## Operations

* [Table Lookups basics](core-features/table_lookups.md)
* [Non-linear operations](core-features/non_linear_operations.md)
* Other operations
  * [Bit extraction](core-features/bit_extraction.md)
  * [Common tips](core-features/workarounds.md)
  * [Extensions](core-features/extensions.md)

## Compilation

* [Combining compiled functions](compilation/combining.md)
   * [With composition](compilation/composition.md)
   * [With modules](compilation/composing_functions_with_modules.md)
* Key-related options for faster execution
   * [Multi precision](compilation/multi_precision.md)
   * [Multi parameters](compilation/multi_parameters.md)
* [Compression](compilation/compression.md)
* [Reusing arguments](compilation/reuse_arguments.md)
* [Parameter compatibility with restrictions](compilation/parameter_compatibility_with_restrictions.md)
* [Common errors](compilation/common_errors.md)

## Execution / Analysis

* [Simulation](execution-analysis/simulation.md)
* [Debugging and artifact](execution-analysis/debug.md)
* [Performance](optimization/summary.md)
* [GPU acceleration](execution-analysis/gpu_acceleration.md)
* [Rust integration](execution-analysis/rust_integration.md)
* Other
  * [Statistics](compilation/statistics.md)
  * [Progressbar](execution-analysis/progressbar.md)
  * [Formatting and drawing](execution-analysis/formatting_and_drawing.md)

## Guides

* [Configure](guides/configure.md)
* [Manage keys](guides/manage_keys.md)
* [Deploy](guides/deploy.md)
* [TFHE-rs Interoperability](guides/tfhers/README.md)
  * [Shared key](guides/tfhers/shared-key.md)
  * [Serialization](guides/tfhers/serialization.md)
* [Optimization](optimization/self.md)
  * [Improve parallelism](optimization/improve-parallelism/self.md)
    * [Dataflow parallelism](optimization/improve-parallelism/dataflow.md)
    * [Tensorizing operations](optimization/improve-parallelism/tensorization.md)
  * [Optimize table lookups](optimization/optimize-table-lookups/self.md)
    * [Reducing TLU](optimization/optimize-table-lookups/reducing-amount.md)
    * [Implementation strategies](optimization/optimize-table-lookups/strategies.md)
    * [Round/truncating](optimization/optimize-table-lookups/round-truncate.md)
    * [Approximate mode](optimization/optimize-table-lookups/approximate.md)
    * [Bit extraction](optimization/optimize-table-lookups/bit-extraction.md)
  * [Optimize cryptographic parameters](optimization/optimize-cryptographic-parameters/self.md)
    * [Error probability](optimization/optimize-cryptographic-parameters/p-error.md)
    * [Composition](optimization/optimize-cryptographic-parameters/composition.md)

## Tutorials

* [See all tutorials](tutorials/see-all-tutorials.md)
* [Part I: Concrete - FHE compiler](https://www.zama.ai/post/zama-concrete-fully-homomorphic-encryption-compiler)
* [Part II: The Architecture of Concrete](https://www.zama.ai/post/the-architecture-of-concrete-zama-fully-homomorphic-encryption-compiler-leveraging-mlir)

## References

* [API](dev/api/README.md)
* [Supported operations](dev/compatibility.md)

## Explanations

* [Compiler workflow](dev/compilation/compiler_workflow.md)
* Advanced features
  * [Table Lookups advanced](core-features/table_lookups_advanced.md)
  * [Rounding](core-features/rounding.md)
  * [Truncating](core-features/truncating.md)
  * [Floating points](core-features/floating_points.md)
  * [Comparisons](core-features/comparisons.md)
  * [Min/Max operations](core-features/minmax.md)
  * [Bitwise operations](core-features/bitwise.md)
  * [Direct circuits](compilation/direct_circuits.md)
  * [Tagging](core-features/tagging.md)
* [Cryptography basics](core-features/fhe_basics.md)
* [Security](explanations/security.md)
* [Frontend fusing](explanations/fusing.md)

## Developers

* [Contributing](dev/contributing.md)
  * [Project layout](explanations/layout.md)
  * [Compiler backend](explanations/backends/README.md)
    * [Adding a new backend](explanations/backends/new_backend.md)
  * [Optimizer](explanations/optimizer.md)
  * [MLIR FHE dialects](explanations/dialects.md)
    * [FHELinalg dialect](explanations/FHELinalgDialect.md)
    * [FHE dialect](explanations/FHEDialect.md)
    * [TFHE dialect](explanations/TFHEDialect.md)
    * [Concrete dialect](explanations/ConcreteDialect.md)
    * [Tracing dialect](explanations/TracingDialect.md)
    * [Runtime dialect](explanations/RTDialect.md)
    * [SDFG dialect](explanations/SDFGDialect.md)
  * [Call FHE circuits from other languages](explanations/call_from_other_language.md)
  * [Benchmarking](dev/benchmarking.md)
  * [Examples](dev/examples.md)
  * [Making a release](explanations/releasing.md)
* [Release note](https://github.com/zama-ai/concrete/releases)
* [Feature request](https://github.com/zama-ai/concrete/issues/new?assignees=\&labels=feature\&projects=\&template=features.md)
* [Bug report](https://github.com/zama-ai/concrete/issues/new?assignees=\&labels=bug%2C+triage\&projects=\&template=bug_report.md)
