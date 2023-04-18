# benchmark_concrete_cuda

This benchmark tool is written over Google Benchmark library. It measures the performance of the concrete framework's CUDA-accelerated functions and helps identify potential bottlenecks. 
The output format can be adjusted according to the user's interest. 

Each benchmark is executed once and targets a single function. Internally, for each benchmark, the tool will repeat each targetted function as many times as it needs to report an execution time with sufficient reliability. At this point, the variation we've observed in the benchmarked functions is relatively small, and thus we chose not to repeat benchmarks by default. However, this can also be tuned by the user if needed.

## How to Compile

The first step in compiling code with CMake is to create a build directory. This directory will 
contain all the files generated during the build process, such as object files and executables. 
We recommend creating this directory outside of the source directory, but inside the 
implementation folder,  to keep the source directory clean.

```bash
$ cd concrete/backends/concrete-cuda/implementation
$ mkdir build
$ cd build
```

Run CMake to generate the build files and then use make to compile the project.

```bash
$ cmake ..
$ make
```

The binary will be found in `concrete/backends/concrete-cuda/implementation/build/test_and_benchmark
/benchmark`.

## How to Run Benchmarks

To run benchmarks, you can simply execute the `benchmark_concrete_cuda` executable with no arguments:

```bash
$ test_and_benchmark/benchmark/benchmark_concrete_cuda
```

This will run all the available benchmarks.

## Output format

The reports will be printed to the standard output if you don't pass any arguments. However, Google Benchmarks has extended documentation on how to print it to files with different formats, e.g., `--benchmark_format=json` will print everything to a JSON file.

## How to Filter Benchmarks

You can filter benchmarks by specifying a regular expression as an argument. Only benchmarks whose name matches the regular expression will be executed.

For example, to run only benchmarks whose name contains the word "Bootstrap", you can execute:

```bash
$ test_and_benchmark/benchmark/benchmark_concrete_cuda --benchmark_filter=Bootstrap
```

The parameter `--benchmark_list_tests` can be used to list all the available benchmarks. 

## How to Set the Time Unit

By default, benchmarks are reported in seconds. However, you can change the time unit to one of the following:

* `ns` (nanoseconds)
* `us` (microseconds)
* `ms` (milliseconds)
* `s` (seconds)

To set the time unit, use the --benchmark_time_unit option followed by the desired time unit:

```bash
$ test_and_benchmark/benchmark/benchmark_concrete_cuda --benchmark_time_unit=us
```

## How to Set the Number of Iterations

By default, each benchmark is executed for a number of iterations that is automatically determined by the Google Benchmark library. 
However, you can increase the minimum time used for each measurement to increase the number of 
iterations by using `--benchmark_min_time`. For instance:

```bash
$ test_and_benchmark/benchmark/benchmark_concrete_cuda --benchmark_min_time=10
```

 will force the tool to run at least 10s of iterations.

## Statistics about the benchmarks

By default each benchmark will be executed only once. However, if you use 
`--benchmark_repetitions` you can increase that and compute the mean, median, and standard 
deviation of the benchmarks. 

```bash
$ test_and_benchmark/benchmark/benchmark_concrete_cuda --benchmark_repetitions=10
```

Doing this, for each run the execution time will be reported. If you prefer, you can use 
`--benchmark_report_aggregates_only=true` to report only the statistical data, or 
`--benchmark_display_aggregates_only=true` that will display in the standard output only the 
statistical data but report everything in the output file. 

## Known issues

When displayed in the standard output, on a terminal, the unit presented for the throughput is given in "number of operations per second". This is a bug on the way data is presented by Google Benchmark. The correct unit is "operations per dollar".

## Conclusion

With these options, you can easily run benchmarks, filter benchmarks, set the time unit, and the number of iterations of benchmark_concrete_cuda. If you have any questions or issues, please feel free to contact us.
To learn more about Google Benchmark library, please refer to the [official user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md). 
