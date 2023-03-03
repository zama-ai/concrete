# Run and display summary

## Run

You can:
- ```make stress-tests```, tests replications are parallelized but tests are not parallelized
- ```make stress-tests-fast```, tests and KeySetCache generation are parallelized, useful for a first run

## Summary

```make show-stress-tests```

# Raw results

In directory ```streestests/trace```:
- ```test_controlled```, contains experiments with controlled code and parameters cases that should run fine
- ```test_wild```, contains experiments with less controlled code and parameters that explores the limits of the compiler.

All experiment are currently a weighted sum with a constant weight followed by an identity function.

These two directories contains one experiment file per experiment, named ```XXXbits_x_YYY_W``` where XXX is the precision, YYY is the size of the computation and W is the experiment non structural parameter (here the weight in the sum).

Files are in json format but can easily be grepped (multi-lines).

# Experiment file

```json
{
  # Command line to relauch an experiment replication by end
  "cmd": "concretecompiler /tmp/stresstests/basic_001_002_1.mlir --action=jit-invoke --jit-funcname=main --jit-args=1 --jit-args=1",
  # General information about the experiment
  "conditions": {
    "bitwidth": 1, # precision in bits
    "size": 2, # size of the computation
    "args": [  # jit arguments
      1,
      1
    ],
    "log_manp_max": 3, # value comuted by concretecompiler
    "overflow": true,  # does the exact computation overflow the precision
    "details": [
      "OVERFLOW"
    ] # message related to potential issues 
  },
  # Replications results
  "replications": [
    {
      "success": true,
      "details": []
    }, # A successful replication
    {
      "success": true,
      "details": [
        "OVERFLOW 3"
      ]
    }, # A successful replication with the overflow value, result being correct when truncated
    {
      "success": false,
      "details": [
        "OVERFLOW 3",
        "Expected: 4 vs. 3 (no modulo 0 vs. 1)"
      ]
    } # A failed replication when the result is wrong both directly and after truncation
    ...
  ],
  "code": "\nfunc.func @main(...) { ... }",
  "success_rate": 99.0,
  "overflow_rate": 100.0
}
```
