#include "concrete-optimizer.hpp"
#include <cassert>
#include <vector>

template <typename T>
rust::cxxbridge1::Slice<const T> slice(std::vector<T> &vec) {
  const T *data = vec.data();

  return rust::cxxbridge1::Slice<const T>(data, vec.size());
}

const uint64_t SECURITY_128B = 128;
const double P_ERROR = 0.05;
const double PRECISION_1B = 1;
const double PRECISION_8B = 8;
const double PRECISION_16B = 16;
const double WOP_FALLBACK_LOG_NORM = 8;
const double NOISE_DEVIATION_COEFF = 1.0;

void test_v0() {
  concrete_optimizer::v0::Solution solution =
      concrete_optimizer::v0::optimize_bootstrap(
          PRECISION_1B, SECURITY_128B, NOISE_DEVIATION_COEFF, P_ERROR);

  assert(solution.glwe_polynomial_size == 1024);
}

void test_dag_no_lut() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex node1 =
      dag->add_input(PRECISION_8B, slice(shape));

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {node1};

  std::vector<uint64_t> weight_vec = {1, 1, 1};

  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights =
      concrete_optimizer::weights::vector(slice(weight_vec));

  concrete_optimizer::dag::OperatorIndex node2 =
      dag->add_dot(slice(inputs), std::move(weights));

  auto solution = dag->optimize_v0(SECURITY_128B, P_ERROR);
  assert(solution.glwe_polynomial_size == 1024);
}

void test_dag_lut() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      dag->add_input(PRECISION_8B, slice(shape));

  std::vector<u_int64_t> table = {};
  concrete_optimizer::dag::OperatorIndex node2 =
      dag->add_lut(input, slice(table), PRECISION_8B);

  auto solution = dag->optimize(SECURITY_128B, P_ERROR, WOP_FALLBACK_LOG_NORM);
  assert(solution.glwe_dimension == 1);
  assert(solution.glwe_polynomial_size == 8192);
  assert(!solution.use_wop_pbs);
}

void test_dag_lut_wop() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      dag->add_input(PRECISION_16B, slice(shape));

  std::vector<u_int64_t> table = {};
  concrete_optimizer::dag::OperatorIndex node2 =
      dag->add_lut(input, slice(table), PRECISION_16B);

  auto solution = dag->optimize(SECURITY_128B, P_ERROR, WOP_FALLBACK_LOG_NORM);
  assert(solution.glwe_dimension == 2);
  assert(solution.glwe_polynomial_size == 1024);
  assert(solution.use_wop_pbs);
}

int main(int argc, char *argv[]) {
  test_v0();
  test_dag_no_lut();
  test_dag_lut();
  test_dag_lut_wop();

  return 0;
}
