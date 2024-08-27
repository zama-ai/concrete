# GPU acceleration

This document explains how to use GPU accelerations with Concrete.

Concrete supports acceleration using one or more GPUs.

{% hint style="info" %}
This version is not available on [pypi.org](https://pypi.org/project/concrete-python), which only hosts wheels with CPU support. 
{% endhint %}

To use GPU acceleration, install the GPU/CUDA wheel from our [Zama public PyPI repository](https://pypi.zama.ai) using the following command:

`pip install concrete-python  --extra-index-url https://pypi.zama.ai/gpu`.

After installing the GPU/CUDA wheel, you must [configure](../guides/configure.md) the FHE program compilation to enable GPU offloading using the `use_gpu` option.

{% hint style="info" %}
Our GPU wheels are built with CUDA 11.8 and should be compatible with higher versions of CUDA.
{% endhint %}

## GPU execution configuration

By default the compiler and runtime will use all available system resources, including all CPU cores and GPUs. You can adjust this by using the following environment variables:

### SDFG_NUM_THREADS
- **Type**: Integer
- **Default value**: The number of hardware threads on the system (including hyperthreading) minus the number of GPUs in use.
- **Description:** This variable determines the number of CPU threads that execute in paralelle with the GPU for offloadable workloads. GPU scheduler threads (including CUDA threads and those used within Concrete) are necessary but can block or interfere with worker thread execution. Therefore, it is recommended to undersubscribe the CPU hardware threads by the number of GPU devices used.


### SDFG_NUM_GPUS
- **Type**: Integer
- **Default value**: The number of GPUs available.
- **Description**: This value determines the number of GPUs to use for offloading. This can be set to any value between 1 and the total number of GPUs on the system.

### SDFG_MAX_BATCH_SIZE**

- **Type**: Integer
- **Default value**: LLONG_MAX (no batch size limit)
- **Description**: This value limits the maximum batch size for offloading in cases where the GPU memory is insufficient.


### SDFG_DEVICE_TO_CORE_RATIO

- **Type**: Integer
- **Default value**: The ratio between the compute capability of the GPU (at index 0) and a CPU core.
- **Description**: This ratio is used to balance the load between the CPU and GPU. If the GPU is underutilized, set this value higher to increase the amount of work offloaded to the GPU.


### OMP_NUM_THREADS

- **Type**: Integer
- **Default value**: The number of hardware threads on the system, including hyperthreading.
- **Description**: This value specifies the portions of program execution that are not yet supported for GPU offload, which will be parallelized using OpenMP on the CPU.

