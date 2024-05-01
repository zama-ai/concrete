# GPU acceleration

Concrete supports acceleration using one or more GPUs. In order to enable this feature, the GPU/CUDA flavor python wheel must be installed and compilation must be [configured](../guides/configure.md) using the **use_gpu** option.

## GPU execution configuration

By default the compiler and runtime will make use of all resources available on the system, to include all CPU cores and GPUs. This can be adjusted by using environment variables.
The following variables are relevant in this context:

* **SDFG_NUM_THREADS**: Integer = Number of hardware threads on the system, including hyperthreading, less number of GPUs in use.
  * Number of CPU threads to execute concurrently to GPU for workloads that can be offloaded. As GPU scheduler threads (including CUDA threads and those used within Concrete) are necessary and can be a bottleneck or interfere with worker thread execution, it is recommended to undersubscribe the CPU hardware threads by the number of GPU devices used.
* **SDFG_NUM_GPUS**: Integer = Number of GPUs available
  * Number of GPUs to use for offloading. This can be set at any value between 1 and the total number of GPUs on the system.
* **SDFG_MAX_BATCH_SIZE**: Integer = LLONG_MAX (no batch size limit)
  * Limit the maximum batch size to offload in cases where the GPU memory is insufficient.
* **SDFG_DEVICE_TO_CORE_RATIO**: Integer = Ratio between the compute capability of the GPU (at index 0) and a CPU core
  * Ratio between GPU and CPU used to balance the load between CPU and GPU. If the GPU is starved, this can be set at higher values to increase the amount of work offloaded.
* **OMP_NUM_THREADS**: Integer = Number of hardware threads on the system, including hyperthreading
  * Portions of program execution that are not yet supported for GPU offload are parallelized using OpenMP on the CPU.
