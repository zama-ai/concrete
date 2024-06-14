# Table of contents

* [Welcome](README.md)

## Get Started

* [What is Concrete?](get-started/readme.md)
* [Installation](get-started/installing.md)
* [Quick start](get-started/quick_start.md)
* [Compatibility](get-started/compatibility.md)
* [Terminology](get-started/terminology.md)

## Core features

* [Overview](core-features/fhe_basics.md)
* [Table lookups (basics)](core-features/table_lookups.md)
* [Bit extraction](core-features/bit_extraction.md)
* [Non-linear operations](core-features/non_linear_operations.md)
* [Common tips](core-features/workarounds.md)
* [Extensions](core-features/extensions.md)
* [Tagging](core-features/tagging.md)

## Compilation

* [Composing functions with modules](compilation/composing_functions_with_modules.md)
* [Compression](compilation/compression.md)
* [Reuse arguments](compilation/reuse_arguments.md)
* [Multi precision](compilation/multi_precision.md)
* [Multi parameters](compilation/multi_parameters.md)
* [Decorator](compilation/decorator.md)
* [Direct circuits](compilation/direct_circuits.md)

## Execution / Analysis

* [Simulation](execution-analysis/simulation.md)
* [Progressbar](execution-analysis/progressbar.md)
* [Statistics](compilation/statistics.md)
* [Formatting and drawing](execution-analysis/formatting_and_drawing.md)
* [Debug](execution-analysis/debug.md)
* [GPU acceleration](execution-analysis/gpu_acceleration.md)

## Guides

* [Configure](guides/configure.md)
* [Manage keys](guides/manage_keys.md)
* [Deploy](guides/deploy.md)

## Tutorials

* [See all tutorials](tutorials/see-all-tutorials.md)
* [Part I: Concrete - FHE compiler](https://www.zama.ai/post/zama-concrete-fully-homomorphic-encryption-compiler)
* [Part II: The Architecture of Concrete](https://www.zama.ai/post/the-architecture-of-concrete-zama-fully-homomorphic-encryption-compiler-leveraging-mlir)

## References

* [API](dev/api/README.md)

## Explanations

* [Compiler workflow](dev/compilation/compiler_workflow.md)
* [Compiler internals](dev/compilation/compiler_internals.md)
  * [Table lookups](core-features/table_lookups_advanced.md)
  * [Rounding](core-features/rounding.md)
  * [Truncating](core-features/truncating.md)
  * [Floating points](core-features/floating_points.md)
  * [Comparisons](core-features/comparisons.md)
  * [Min/Max operations](core-features/minmax.md)
  * [Bitwise operations](core-features/bitwise.md)
* [Frontend fusing](explanations/fusing.md)
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
* [Security](explanations/security.md)
* [Call FHE circuits from other languages](explanations/call_from_other_language.md)
* [Project layout](explanations/layout.md)

## Developers

* [Contributing](dev/contributing.md)
* [Release note](https://github.com/zama-ai/concrete/releases)
* [Feature request](https://github.com/zama-ai/concrete/issues/new?assignees=\&labels=feature\&projects=\&template=features.md)
* [Bug report](https://github.com/zama-ai/concrete/issues/new?assignees=\&labels=bug%2C+triage\&projects=\&template=bug_report.md)
