// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/debug_probes.h"
#include <cassert>

namespace mlir {
namespace concretelang {
namespace debug {

ProbeBuffer &ProbeBuffer::instance() {
  static ProbeBuffer buf;
  return buf;
}

void ProbeBuffer::reset() {
  std::lock_guard<std::mutex> lock(mutex);
  entries.clear();
}

void ProbeBuffer::record_plaintext(uint32_t probe_id, int64_t value,
                                   const char *tag_ptr, uint32_t tag_len,
                                   uint32_t nmsb) {
  std::lock_guard<std::mutex> lock(mutex);
  entries.push_back(
      {probe_id, std::string(tag_ptr, tag_len), value, nmsb});
}

size_t ProbeBuffer::size() const { return entries.size(); }

const ProbeEntry &ProbeBuffer::get(size_t index) const {
  assert(index < entries.size());
  return entries[index];
}

const std::vector<ProbeEntry> &ProbeBuffer::all() const { return entries; }

} // namespace debug
} // namespace concretelang
} // namespace mlir

extern "C" {

void memref_debug_probe_plaintext(int64_t value, int64_t input_width,
                                  int32_t probe_id, char *tag_ptr,
                                  int32_t tag_len, int32_t nmsb) {
  mlir::concretelang::debug::ProbeBuffer::instance().record_plaintext(
      static_cast<uint32_t>(probe_id), value,
      tag_ptr, static_cast<uint32_t>(tag_len),
      static_cast<uint32_t>(nmsb));
}

void debug_probe_buffer_reset() {
  mlir::concretelang::debug::ProbeBuffer::instance().reset();
}

uint64_t debug_probe_buffer_size() {
  return static_cast<uint64_t>(
      mlir::concretelang::debug::ProbeBuffer::instance().size());
}
}
