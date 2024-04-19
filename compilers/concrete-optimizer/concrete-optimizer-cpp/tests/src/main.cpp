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
const uint32_t CIPHERTEXT_MODULUS_LOG = 64;

concrete_optimizer::Options default_options() {
  return concrete_optimizer::Options{
      .security_level = SECURITY_128B,
      .maximum_acceptable_error_probability = P_ERROR,
      .key_sharing = false,
      .multi_param_strategy = concrete_optimizer::MultiParamStrategy::ByPrecision,
      .default_log_norm2_woppbs = WOP_FALLBACK_LOG_NORM,
      .use_gpu_constraints = false,
      .encoding = concrete_optimizer::Encoding::Auto,
      .cache_on_disk = true,
      .ciphertext_modulus_log = CIPHERTEXT_MODULUS_LOG,
      .fft_precision = 53,
  };
}

#define TEST static void

TEST test_v0() {
  auto options = default_options();
  concrete_optimizer::v0::Solution solution =
      concrete_optimizer::v0::optimize_bootstrap(
          PRECISION_1B, NOISE_DEVIATION_COEFF, options);

  assert(solution.glwe_polynomial_size == 256);
}

TEST test_dag_no_lut() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex node1 =
      builder->add_input(PRECISION_8B, slice(shape));

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {node1};

  std::vector<int64_t> weight_vec = {1, 1, 1};

  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights =
      concrete_optimizer::weights::vector(slice(weight_vec));

  auto id = builder->add_dot(slice(inputs), std::move(weights));
  builder->tag_operator_as_output(id);

  auto solution = dag->optimize(default_options());
  assert(solution.glwe_polynomial_size == 1);
  assert(solution.glwe_dimension == 555);
}

TEST test_dag_lut() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      builder->add_input(PRECISION_8B, slice(shape));

  std::vector<u_int64_t> table = {};
  auto id = builder->add_lut(input, slice(table), PRECISION_8B);
  builder->tag_operator_as_output(id);

  auto solution = dag->optimize(default_options());
  assert(solution.glwe_dimension == 1);
  assert(solution.glwe_polynomial_size == 8192);
  assert(!solution.use_wop_pbs);
}

TEST test_dag_lut_wop() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      builder->add_input(PRECISION_16B, slice(shape));

  std::vector<u_int64_t> table = {};
  auto id = builder->add_lut(input, slice(table), PRECISION_16B);
  builder->tag_operator_as_output(id);

  auto solution = dag->optimize(default_options());
  assert(solution.glwe_dimension == 2);
  assert(solution.glwe_polynomial_size == 1024);
  assert(solution.use_wop_pbs);
}

TEST test_dag_lut_force_wop() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      builder->add_input(PRECISION_8B, slice(shape));

  std::vector<u_int64_t> table = {};
  auto id = builder->add_lut(input, slice(table), PRECISION_8B);
  builder->tag_operator_as_output(id);

  auto options = default_options();
  options.encoding = concrete_optimizer::Encoding::Crt;
  auto solution = dag->optimize(options);
  assert(solution.use_wop_pbs);
  assert(!solution.crt_decomposition.empty());
}

TEST test_multi_parameters_1_precision() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input =
      builder->add_input(PRECISION_8B, slice(shape));

  std::vector<u_int64_t> table = {};
  auto id = builder->add_lut(input, slice(table), PRECISION_8B);
  builder->tag_operator_as_output(id);

  auto options = default_options();
  auto circuit_solution = dag->optimize_multi(options);
  assert(circuit_solution.is_feasible);
  auto secret_keys = circuit_solution.circuit_keys.secret_keys;
  assert(circuit_solution.circuit_keys.secret_keys.size() == 2);
  assert(circuit_solution.circuit_keys.secret_keys[0].identifier == 0);
  assert(circuit_solution.circuit_keys.secret_keys[1].identifier == 1);
  assert(circuit_solution.circuit_keys.bootstrap_keys.size() == 1);
  assert(circuit_solution.circuit_keys.keyswitch_keys.size() == 1);
  assert(circuit_solution.circuit_keys.keyswitch_keys[0].identifier == 0);
  assert(circuit_solution.circuit_keys.keyswitch_keys[0].identifier == 0);
  assert(circuit_solution.circuit_keys.conversion_keyswitch_keys.size() == 0);
}

TEST test_multi_parameters_2_precision() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input1 =
      builder->add_input(PRECISION_8B, slice(shape));

  concrete_optimizer::dag::OperatorIndex input2 =
      builder->add_input(PRECISION_1B, slice(shape));

  std::vector<u_int64_t> table = {};
  auto lut1 = builder->add_lut(input1, slice(table), PRECISION_8B);
  auto lut2 = builder->add_lut(input2, slice(table), PRECISION_8B);

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {lut1, lut2};

  std::vector<int64_t> weight_vec = {1, 1};

  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights =
      concrete_optimizer::weights::vector(slice(weight_vec));

  auto id = builder->add_dot(slice(inputs), std::move(weights));
  builder->tag_operator_as_output(id);

  auto options = default_options();
  auto circuit_solution = dag->optimize_multi(options);
  assert(circuit_solution.is_feasible);
  auto secret_keys = circuit_solution.circuit_keys.secret_keys;
  assert(circuit_solution.circuit_keys.secret_keys.size() == 4);
  assert(circuit_solution.circuit_keys.bootstrap_keys.size() == 2);
  assert(circuit_solution.circuit_keys.keyswitch_keys.size() ==
         2); // 1 layer so less ks
  std::string actual =
      circuit_solution.circuit_keys.conversion_keyswitch_keys[0]
          .description.c_str();
  std::string expected = "fks[1->0]";
  assert(actual == expected);
}

TEST test_multi_parameters_2_precision_crt() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {3};

  concrete_optimizer::dag::OperatorIndex input1 =
      builder->add_input(PRECISION_8B, slice(shape));

  concrete_optimizer::dag::OperatorIndex input2 =
      builder->add_input(PRECISION_1B, slice(shape));

  std::vector<u_int64_t> table = {};
  auto lut1 = builder->add_lut(input1, slice(table), PRECISION_8B);
  auto lut2 = builder->add_lut(input2, slice(table), PRECISION_8B);

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {lut1, lut2};

  std::vector<int64_t> weight_vec = {1, 1};

  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights =
      concrete_optimizer::weights::vector(slice(weight_vec));

  auto id = builder->add_dot(slice(inputs), std::move(weights));
  builder->tag_operator_as_output(id);

  auto options = default_options();
  options.encoding = concrete_optimizer::Encoding::Crt;
  auto circuit_solution = dag->optimize_multi(options);
  assert(circuit_solution.is_feasible);
  auto secret_keys = circuit_solution.circuit_keys.secret_keys;
  assert(circuit_solution.circuit_keys.secret_keys.size() == 2);
  assert(circuit_solution.circuit_keys.bootstrap_keys.size() == 1);
  assert(circuit_solution.circuit_keys.keyswitch_keys.size() == 1);
  assert(circuit_solution.circuit_keys.conversion_keyswitch_keys.size() == 0);
}

TEST test_composable_dag_mono_fallback_on_dag_multi() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {};

  concrete_optimizer::dag::OperatorIndex input1 =
      builder->add_input(PRECISION_8B, slice(shape));

  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {input1};
  std::vector<int64_t> weight_vec = {1 << 8};
  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights1 =
    concrete_optimizer::weights::vector(slice(weight_vec));

  input1 = builder->add_dot(slice(inputs), std::move(weights1));
  std::vector<u_int64_t> table = {};
  auto lut1 = builder->add_lut(input1, slice(table), PRECISION_8B);
  std::vector<concrete_optimizer::dag::OperatorIndex> lut1v = {lut1};
  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights2 =
    concrete_optimizer::weights::vector(slice(weight_vec));
  auto id = builder->add_dot(slice(lut1v), std::move(weights2));
  builder->tag_operator_as_output(id);

  auto options = default_options();
  auto solution1 = dag->optimize(options);
  assert(!solution1.use_wop_pbs);
  assert(solution1.p_error < options.maximum_acceptable_error_probability);

  dag->add_all_compositions();
  auto solution2 = dag->optimize(options);
  assert(!solution2.use_wop_pbs);
  assert(solution2.p_error < options.maximum_acceptable_error_probability);
  assert(solution1.complexity < solution2.complexity);
}

TEST test_non_composable_dag_mono_fallback_on_woppbs() {
  auto dag = concrete_optimizer::dag::empty();
  auto builder = dag->builder("test");

  std::vector<uint64_t> shape = {};

  concrete_optimizer::dag::OperatorIndex input1 =
      builder->add_input(PRECISION_8B, slice(shape));


  std::vector<concrete_optimizer::dag::OperatorIndex> inputs = {input1};
  std::vector<int64_t> weight_vec = {1 << 16};
  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights1 =
    concrete_optimizer::weights::vector(slice(weight_vec));

  input1 = builder->add_dot(slice(inputs), std::move(weights1));
  std::vector<u_int64_t> table = {};
  auto lut1 = builder->add_lut(input1, slice(table), PRECISION_8B);
  std::vector<concrete_optimizer::dag::OperatorIndex> lut1v = {lut1};
  rust::cxxbridge1::Box<concrete_optimizer::Weights> weights2 =
    concrete_optimizer::weights::vector(slice(weight_vec));
  auto id = builder->add_dot(slice(lut1v), std::move(weights2));
  builder->tag_operator_as_output(id);

  auto options = default_options();

  auto solution1 = dag->optimize(options);
  assert(!solution1.use_wop_pbs);
  assert(solution1.p_error < options.maximum_acceptable_error_probability);

  dag->add_all_compositions();
  auto solution2 = dag->optimize(options);
  assert(solution2.p_error < options.maximum_acceptable_error_probability);
  assert(solution1.complexity < solution2.complexity);
  assert(solution2.use_wop_pbs);

}

int main() {

  test_v0();
  test_dag_no_lut();
  test_dag_lut();
  test_dag_lut_wop();
  test_dag_lut_force_wop();
  test_multi_parameters_1_precision();
  test_multi_parameters_2_precision();
  test_multi_parameters_2_precision_crt();
  test_composable_dag_mono_fallback_on_dag_multi();
  test_non_composable_dag_mono_fallback_on_woppbs();

  return 0;
}
