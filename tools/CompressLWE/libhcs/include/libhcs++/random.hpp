/**
 * @file hcs_random.hpp
 *
 * C++ wrapper class for the hcs_random object.
 */

#ifndef HCS_RANDOM_HPP
#define HCS_RANDOM_HPP

#include "../libhcs/hcs_random.h"

namespace hcs {

class random {

private:
  hcs_random *hr;

public:
  random(const random &) = delete;
  random() { hr = hcs_init_random(); }

  ~random() { hcs_free_random(hr); }

  int reseed() { return hcs_reseed_random(hr); }

  hcs_random *as_ptr() { return hr; }
};

} // namespace hcs

#endif
