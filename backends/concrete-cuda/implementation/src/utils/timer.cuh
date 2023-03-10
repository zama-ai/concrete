#ifndef CNCRT_TIMER_H
#define CNCRT_TIMER_H

#include <iostream>
#define synchronize_threads_in_block() __syncthreads()

template <bool active> class CudaMeasureExecution {
public:
  cudaEvent_t m_start, m_stop;

  __host__ CudaMeasureExecution() {
    if constexpr (active) {
      cudaEventCreate(&m_start);
      cudaEventCreate(&m_stop);
      cudaEventRecord(m_start);
    }
  }

  __host__ ~CudaMeasureExecution() {
    if constexpr (active) {
      float ms;
      cudaEventRecord(m_stop);
      cudaEventSynchronize(m_stop);
      cudaEventElapsedTime(&ms, m_start, m_stop);
      std::cout << "Execution took " << ms << "ms" << std::endl;
    }
  }
};

#endif // CNCRT_TIMER_H