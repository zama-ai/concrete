#include "concrete-optimizer.hpp"
#include <vector>

template <typename T>
rust::cxxbridge1::Slice<const T> slice(std::vector<T> &vec) {
  const T *data = vec.data();

  return rust::cxxbridge1::Slice<const T>(data, vec.size());
}

int test1() {
  concrete_optimizer::v0::Solution solution =
      concrete_optimizer::v0::optimize_bootstrap(1, 128, 1, .05);

  if (solution.glwe_polynomial_size != 1024) {
    return 1;
  }

  return 0;
}

int test2() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex node1 =
      dag->add_input(1, slice(shape));

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {node1};

  std::vector<uint64_t> weight_vec = {3};

  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights =
      concrete_optimizer::weights::vector(slice(weight_vec));

  concrete_optimizer::dag::OperatorIndex node2 =
      dag->add_dot(slice(inputs), std::move(weights));

  return 0;
}

int main(int argc, char *argv[]) {
  int err = test1();

  if (err)
    return err;

  err = test2();

  if (err)
    return err;

  return 0;
}
