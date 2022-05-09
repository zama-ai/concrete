#include "concrete-optimizer.hpp"

int main(int argc, char *argv[]) {
  concrete_optimizer::Solution solution = concrete_optimizer::optimise_bootstrap(1, 128, 1, .05);

  if (solution.glwe_polynomial_size != 1024) {
    return 1;
  }

  return 0;
}
