// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

/**
   Define the API exposed to the compiler for code generation.
 */

#ifndef CONCRETELANG_DFR_RUNTIME_API_H
#define CONCRETELANG_DFR_RUNTIME_API_H
#include <cstddef>
#include <cstdlib>

extern "C" {

typedef void (*wfnptr)(...);

void *_dfr_make_ready_future(void *);
void _dfr_create_async_task(wfnptr, size_t, size_t, ...);
void *_dfr_await_future(void *);

/*  Keys can have node-local copies which can be retrieved.  This
    should only be called on the node where the key is required.  */
void _dfr_register_key(void *, size_t, size_t);
void _dfr_broadcast_keys();
void *_dfr_get_key(size_t);

/*  Memory management:
    _dfr_make_ready_future allocates the future, not the underlying storage.
    _dfr_create_async_task allocates both future and storage for outputs.  */
void _dfr_deallocate_future(void *);
void _dfr_deallocate_future_data(void *);

/*  Initialisation & termination.  */
void _dfr_start();
void _dfr_stop();

void _dfr_pre_main();
void _dfr_post_main();
}
#endif
