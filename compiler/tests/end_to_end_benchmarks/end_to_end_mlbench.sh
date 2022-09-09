#!/bin/bash

find $1 -name "*mlbench_*.yaml" -exec bash -c "BENCHMARK_FILE={} BENCHMARK_STACK=1000000000 BENCHMARK_NAME=MLBench $2" \;
