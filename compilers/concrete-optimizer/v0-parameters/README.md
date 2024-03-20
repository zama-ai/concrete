# v0 Parameters

The `v0-parameters` tool provides crypto-parameters that guarantee security, correctness and fast
computation without prior knowledge of the crypto-parameter optimization.

For a given (`precision`, `log2(norm2)`), these parameters can be used in a TFHE integer circuit where the maximal integer precision is `precision` and the maximal norm2 between table lookups is `2^log2(norm2)`.
The norm2 is the sum of the square of weights in multisum between table lookups or graph inputs (weights on the same input must first be combined as a single weight).
The probability of error is the maximal acceptable probability of error of each table lookup.

It can also be used to explore the crypto-parameter space w.r.t. the 2-norm of the dot product, the
precision or even the failure probability.

For now, we only support two kind of atomic patterns but more will be added in the near future.

## Supported Atomic Patterns

### Default Atomic Pattern

The default atomic pattern is composed of a dot product between ciphertexts and integer weights, an
LWE-to-LWE keyswitch and a PBS i.e.

<div style="text-align: center;"> DotProduct(v) ➜ KS ➜ PBS </div>

This atomic pattern allows to compute over encrypted data a dot product and a lookup-table
evaluation for precision between 1 and 8 (with the default failure probability) and 2-norm between
2^0 and 2^25. The 2-norm is defined as the 2-norm v of the weights of the Dot Product. It is used as
a metric to quantify the impact of the leveled operations between two PBS on the noise  (here a
DotProduct). Note that an atomic pattern can be described by its precision and 2-norm but only in
the case that every inputs are independents from one another (regarding the noise).

### New Atomic Pattern

Another atomic pattern is available leveraging the new WoP-PBS (Without-Padding Programmable
Bootstrapping) described in this [paper](https://eprint.iacr.org/2022/704.pdf). It is composed of a
dot product between ciphertexts and integer weights and a WoP-PBS.

<div style="text-align: center;"> DotProduct(v) ➜ WoP-PBS </div>

Using this new AP, we can find parameters for precision up to 16 bits 🥳 using the optional flag `
--wop-pbs` like that

```bash
cargo run --release -- --wop-pbs
```

## Usage

The `v0-parameters` tool can take several parameters as arguments. The summary of all available
arguments is accessible by running in the `v0-parameters/` folder

```bash
cargo run --release -- --help
```

As an alternative you can use the `optimizer` script in the root directory:

```bash
./optimizer --help
```

By default, the optimization is done on the default AP (DotProduct -> Ks -> PBS) for every available
precision and for every 2-norm. If not specified, the correctness of the computation is guaranteed
up to a failure probability of 2^-13.9. This can be changed using the `--p-error`
optional argument.

## Advanced Usage

### Playing with search spaces

It is possible to choose the search space for each cryptographic parameters. For example, here we
constrain the glwe dimension to be equal to 1:

```bash
./optimizer --max-glwe-dim 1
```

### Generating reference files

Some of our tests are comparing parameters found by previous version of `concrete-optimizer` against
the parameters found by the current state of `concrete-optimizer`.

To generate those references, you must be in v0-parameters directory.
For the default AP you can do:

```bash
cargo run --release --bin v0-parameters-by-level
```

and for the new WoP-PBS AP

```bash
cargo run --release --bin v0-parameters-by-level -- --wop-pbs
```

The reference files will be written in the `ref/` folder.
