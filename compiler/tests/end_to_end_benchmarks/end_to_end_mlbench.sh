#!/bin/bash

if [[ -d $1 ]]; then
  # Execute all generated YAML files sequentially.
  find "$1" -name "*mlbench_*.yaml" -exec bash -c "BENCHMARK_FILE={} BENCHMARK_STACK=1000000000 BENCHMARK_NAME=MLBench $2" \;
else
  # Execute only one of the YAML file.
  bash -c "BENCHMARK_FILE=$1 BENCHMARK_STACK=1000000000 BENCHMARK_NAME=MLBench $2" \;
fi
