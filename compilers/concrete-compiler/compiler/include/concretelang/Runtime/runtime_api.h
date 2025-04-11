// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

/// Define the API exposed to the compiler for code generation.

#ifndef CONCRETELANG_DFR_RUNTIME_API_H
#define CONCRETELANG_DFR_RUNTIME_API_H
#include <cstddef>
#include <cstdlib>

extern "C" {

typedef void (*wfnptr)(...);

void *_dfr_make_ready_future(void *, size_t);
void _dfr_create_async_task(wfnptr, void *, size_t, size_t, ...);
void _dfr_register_work_function(wfnptr);
void *_dfr_await_future(void *);

/*  Memory management:
    _dfr_make_ready_future allocates the future, not the underlying storage.
    _dfr_create_async_task allocates both future and storage for outputs.  */
void _dfr_deallocate_future(void *);

/*  Initialisation & termination.  */
void _dfr_start(int64_t, bool, void *);
void _dfr_stop(int64_t);

void _dfr_terminate();
}

#endif
