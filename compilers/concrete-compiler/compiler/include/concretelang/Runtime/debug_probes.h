// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_DEBUG_PROBES_H
#define CONCRETELANG_RUNTIME_DEBUG_PROBES_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mlir {
namespace concretelang {
namespace debug {

struct ProbeEntry {
  uint32_t probe_id;
  std::string tag;
  int64_t value;
  uint32_t nmsb;
};

class ProbeBuffer {
  std::mutex mutex;
  std::vector<ProbeEntry> entries;

public:
  static ProbeBuffer &instance();

  void reset();

  void record_plaintext(uint32_t probe_id, int64_t value, const char *tag_ptr,
                        uint32_t tag_len, uint32_t nmsb);

  size_t size() const;

  const ProbeEntry &get(size_t index) const;

  const std::vector<ProbeEntry> &all() const;
};

} // namespace debug
} // namespace concretelang
} // namespace mlir

extern "C" {
void memref_debug_probe_plaintext(int64_t value, int64_t input_width,
                                  int32_t probe_id, char *tag_ptr,
                                  int32_t tag_len, int32_t nmsb);

void debug_probe_buffer_reset();

uint64_t debug_probe_buffer_size();
}

#endif
