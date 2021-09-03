#include "zamalang/Runtime/wrappers.h"
#include <stdio.h>

ForeignPlaintextList_u64 *
runtime_foreign_plaintext_list_u64(int *err, uint64_t *allocated,
                                   uint64_t *aligned, uint64_t offset,
                                   uint64_t size_dim0, uint64_t stride_dim0,
                                   uint64_t size, uint32_t precision) {
  if (stride_dim0 != 1) {
    fprintf(stderr, "Runtime: stride not equal to 1, check "
                    "runtime_foreign_plaintext_list_u64");
  }
  // Encode table values in u64
  uint64_t *encoded_table = malloc(size * sizeof(uint64_t));
  for (uint64_t i = 0; i < size; i++) {
    encoded_table[i] = (aligned + offset)[i] << (64 - precision - 1);
  }
  return foreign_plaintext_list_u64(err, encoded_table, size);
  // TODO: is it safe to free after creating plaintext_list?
}
