# benchmark_concrete_cuda

This benchmark tool is written over Google Benchmark library. It measures the performance of the 
CUDA-accelerated functions of the concrete-framework and helps to identify potential 
bottlenecks.

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

## How to Run Benchmarks

To run benchmarks, you can simply execute the `benchmark_concrete_cuda` executable with no arguments:

```bash
$ benchmark/benchmark_concrete_cuda
```

This will run all the benchmarks in the code.

## How to Filter Benchmarks

You can filter benchmarks by specifying a regular expression as an argument. Only benchmarks whose name matches the regular expression will be executed.

For example, to run only benchmarks whose name contains the word "Bootstrap", you can execute:

```bash
$ benchmark/benchmark_concrete_cuda --benchmark_filter=Bootstrap
```

## How to Set the Time Unit

By default, benchmarks are reported in seconds. However, you can change the time unit to one of the following:

* `ns` (nanoseconds)
* `us` (microseconds)
* `ms` (milliseconds)
* `s` (seconds)

To set the time unit, use the --benchmark_time_unit option followed by the desired time unit:

```bash
$ benchmark/benchmark_concrete_cuda --benchmark_time_unit=us
```

## How to Set the Number of Iterations

By default, each benchmark is executed for a number of iterations that is automatically determined by the Google Benchmark library. 
However, you can increase the minimum time used for each measurement to increase the number of 
iterations by using --benchmark_min_time. For instance:

```bash
$ benchmark/benchmark_concrete_cuda --benchmark_min_time=10
```

 will force the tool to run at least 10s of iterations.


## Conclusion

With these options, you can easily run benchmarks, filter benchmarks, set the time unit, and the number of iterations of benchmark_concrete_cuda. If you have any questions or issues, please feel free to contact us.
To learn more about Google Benchmark library, please refer to the [official user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md). 
