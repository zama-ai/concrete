#include "zamalang/Runtime/wrappers.h"
#include <stdio.h>

ForeignPlaintextList_u64 *runtime_foreign_plaintext_list_u64(
    int *err, uint64_t *allocated, uint64_t *aligned, uint64_t offset,
    uint64_t size_dim0, uint64_t stride_dim0, uint64_t size) {
  if (stride_dim0 != 1) {
    fprintf(stderr, "Runtime: stride not equal to 1, check "
                    "runtime_foreign_plaintext_list_u64");
  }
  return foreign_plaintext_list_u64(err, aligned + offset, size);
}
