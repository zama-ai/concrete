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

concrete_optimizer::Options default_options() {
  return concrete_optimizer::Options {
    .security_level = SECURITY_128B,
    .maximum_acceptable_error_probability = P_ERROR,
    .default_log_norm2_woppbs = WOP_FALLBACK_LOG_NORM,
    .use_gpu_constraints = false,
    .encoding = concrete_optimizer::Encoding::Auto,
    .cache_on_disk = true,
  };
}

void test_v0() {
  concrete_optimizer::v0::Solution solution =
      concrete_optimizer::v0::optimize_bootstrap(
          PRECISION_1B, NOISE_DEVIATION_COEFF, default_options());

  assert(solution.glwe_polynomial_size == 256);
}

void test_dag_no_lut() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex node1 =
      dag->add_input(PRECISION_8B, slice(shape));

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {node1};

  std::vector<int64_t> weight_vec = {1, 1, 1};

  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights =
      concrete_optimizer::weights::vector(slice(weight_vec));

  dag->add_dot(slice(inputs), std::move(weights));

  auto solution = dag->optimize_v0(default_options());
  assert(solution.glwe_polynomial_size == 256);
}

void test_dag_lut() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      dag->add_input(PRECISION_8B, slice(shape));

  std::vector<u_int64_t> table = {};
  dag->add_lut(input, slice(table), PRECISION_8B);

  auto solution = dag->optimize(default_options());
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
  dag->add_lut(input, slice(table), PRECISION_16B);

  auto solution = dag->optimize(default_options());
  assert(solution.glwe_dimension == 2);
  assert(solution.glwe_polynomial_size == 1024);
  assert(solution.use_wop_pbs);
}

void test_dag_lut_force_wop() {
  auto dag = concrete_optimizer::dag::empty();

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      dag->add_input(PRECISION_8B, slice(shape));

  std::vector<u_int64_t> table = {};
  dag->add_lut(input, slice(table), PRECISION_8B);

  auto options = default_options();
  options.encoding = concrete_optimizer::Encoding::Crt;
  auto solution = dag->optimize(options);
  assert(solution.use_wop_pbs);
}

int main() {
  test_v0();
  test_dag_no_lut();
  test_dag_lut();
  test_dag_lut_wop();
  test_dag_lut_force_wop();

  return 0;
}
