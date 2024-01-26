#include <gtest/gtest.h>

#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/Transforms/Transforms.h"

std::string transform(std::string source,
                      concrete_optimizer::dag::CircuitSolution solution) {
  // Register dialect
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::concretelang::TFHE::TFHEDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext mlirContext;
  mlirContext.appendDialectRegistry(registry);

  // Parse from string
  auto memoryBuffer = llvm::MemoryBuffer::getMemBuffer(source);
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> mlirModuleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(sm, &mlirContext);

  // Apply the parametrization pass
  mlir::PassManager pm(&mlirContext);

  pm.addPass(mlir::concretelang::createTFHECircuitSolutionParametrizationPass(
      solution));

  pm.addPass(mlir::createCanonicalizerPass());

  assert(pm.run(mlirModuleRef->getOperation()).succeeded() &&
         "pass manager fail");

  std::string moduleString;
  llvm::raw_string_ostream s(moduleString);

  mlirModuleRef->getOperation()->print(s);

  return s.str();
}

// Returns the secret key id
int addSecretKey(concrete_optimizer::dag::CircuitSolution &solution,
                 int glwe_dimension, int polynomial_size) {
  concrete_optimizer::dag::SecretLweKey secretLweKey;
  secretLweKey.description = "single_key";
  secretLweKey.identifier = 0;
  secretLweKey.glwe_dimension = glwe_dimension;
  secretLweKey.polynomial_size = polynomial_size;
  secretLweKey.identifier = solution.circuit_keys.secret_keys.size();
  solution.circuit_keys.secret_keys.push_back(secretLweKey);
  return secretLweKey.identifier;
}

// Returns the keyswitch key id
int addKeyswitchKey(concrete_optimizer::dag::CircuitSolution &solution,
                    int input_sk, int output_sk, int level, int base_log) {
  concrete_optimizer::dag::KeySwitchKey keySwitchKey;
  keySwitchKey.input_key = solution.circuit_keys.secret_keys[input_sk];
  keySwitchKey.output_key = solution.circuit_keys.secret_keys[output_sk];

  keySwitchKey.ks_decomposition_parameter.level = level;
  keySwitchKey.ks_decomposition_parameter.log2_base = base_log;
  keySwitchKey.identifier = solution.circuit_keys.keyswitch_keys.size();
  solution.circuit_keys.keyswitch_keys.push_back(keySwitchKey);
  return keySwitchKey.identifier;
}

int addExtraKeyswitchKey(concrete_optimizer::dag::CircuitSolution &solution,
                         int input_sk, int output_sk, int level, int base_log) {
  concrete_optimizer::dag::ConversionKeySwitchKey keySwitchKey;
  keySwitchKey.input_key = solution.circuit_keys.secret_keys[input_sk];
  keySwitchKey.output_key = solution.circuit_keys.secret_keys[output_sk];

  keySwitchKey.ks_decomposition_parameter.level = level;
  keySwitchKey.ks_decomposition_parameter.log2_base = base_log;
  keySwitchKey.identifier = solution.circuit_keys.keyswitch_keys.size();
  solution.circuit_keys.conversion_keyswitch_keys.push_back(keySwitchKey);
  return keySwitchKey.identifier;
}

// Returns the bootstrap key id
int addBootstrapKey(concrete_optimizer::dag::CircuitSolution &solution,
                    int input_sk, int output_sk, int level, int base_log) {
  concrete_optimizer::dag::BootstrapKey bootstrapKey;
  // TODO: Interface design identifier or key
  bootstrapKey.input_key = solution.circuit_keys.secret_keys[input_sk];
  bootstrapKey.output_key = solution.circuit_keys.secret_keys[output_sk];
  bootstrapKey.br_decomposition_parameter.level = level;
  bootstrapKey.br_decomposition_parameter.log2_base = base_log;
  bootstrapKey.identifier = solution.circuit_keys.bootstrap_keys.size();
  solution.circuit_keys.bootstrap_keys.push_back(bootstrapKey);
  return bootstrapKey.identifier;
}

void addInstructionKey(concrete_optimizer::dag::CircuitSolution &solution,
                       int input_key, int output_key, int ksk = -1,
                       int bsk = -1,
                       std::vector<uint64_t> extra_conversion_keys = {}) {
  concrete_optimizer::dag::InstructionKeys instrKey;
  instrKey.input_key = input_key;
  instrKey.output_key = output_key;
  instrKey.tlu_bootstrap_key = bsk;
  instrKey.tlu_keyswitch_key = ksk;
  for (const auto &item : extra_conversion_keys) {
    instrKey.extra_conversion_keys.push_back(item);
  }
  solution.instructions_keys.push_back(instrKey);
}

TEST(TFHECircuitParametrization, single_sk) {
  std::string source = R"(
  func.func @main(%arg0: !TFHE.glwe<sk?> {TFHE.OId = 0 : i32}, %arg1: !TFHE.glwe<sk?> {TFHE.OId = 1 : i32}, %arg2: i64) -> !TFHE.glwe<sk?> {
    %0 = "TFHE.add_glwe_int"(%arg0, %arg2) {TFHE.OId = 2 : i32} : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
    %1 = "TFHE.add_glwe"(%0, %arg1) {TFHE.OId = 3 : i32} : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
    return %1 : !TFHE.glwe<sk?>
  }
)";
  std::string expected = R"(module {
  func.func @main(%arg0: !TFHE.glwe<sk<0,1,1024>>, %arg1: !TFHE.glwe<sk<0,1,1024>>, %arg2: i64) -> !TFHE.glwe<sk<0,1,1024>> {
    %0 = "TFHE.add_glwe_int"(%arg0, %arg2) : (!TFHE.glwe<sk<0,1,1024>>, i64) -> !TFHE.glwe<sk<0,1,1024>>
    %1 = "TFHE.add_glwe"(%0, %arg1) : (!TFHE.glwe<sk<0,1,1024>>, !TFHE.glwe<sk<0,1,1024>>) -> !TFHE.glwe<sk<0,1,1024>>
    return %1 : !TFHE.glwe<sk<0,1,1024>>
  }
}
)";
  // TODO: concrete_optimizer::dag::CircuitSolution
  concrete_optimizer::dag::CircuitSolution solution;
  auto keyId = addSecretKey(solution, 1, 1024);
  concrete_optimizer::dag::InstructionKeys instr0;
  // %arg0
  addInstructionKey(solution, keyId, keyId);
  // %arg1
  addInstructionKey(solution, keyId, keyId);
  // %0
  addInstructionKey(solution, keyId, keyId);
  // %1
  addInstructionKey(solution, keyId, keyId);
  std::string output = transform(source, solution);
  ASSERT_EQ(output, expected);
}

TEST(TFHECircuitParametrization, keyswitch) {
  std::string source = R"(
  func.func @main(%arg0: !TFHE.glwe<sk?> {TFHE.OId = 0 : i32}) -> !TFHE.glwe<sk?> {
    %0 = "TFHE.keyswitch_glwe"(%arg0) {TFHE.OId = 1 : i32, key = #TFHE.ksk<sk?, sk?, -1, -1>} : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
    return %0 : !TFHE.glwe<sk?>
  }
)";
  std::string expected = R"(module {
  func.func @main(%arg0: !TFHE.glwe<sk<0,1,1024>>) -> !TFHE.glwe<sk<1,1,1701>> {
    %0 = "TFHE.keyswitch_glwe"(%arg0) {key = #TFHE.ksk<sk<0,1,1024>, sk<1,1,1701>, 2, 12>} : (!TFHE.glwe<sk<0,1,1024>>) -> !TFHE.glwe<sk<1,1,1701>>
    return %0 : !TFHE.glwe<sk<1,1,1701>>
  }
}
)";
  concrete_optimizer::dag::CircuitSolution solution;
  // Add a first secret key
  auto sk0 = addSecretKey(solution, 1, 1024);
  auto sk1 = addSecretKey(solution, 3, 567);
  auto ksk = addKeyswitchKey(solution, sk0, sk1, 2, 12);
  // %arg0
  addInstructionKey(solution, sk0, sk0);
  // %0
  addInstructionKey(solution, sk0, sk1, ksk);
  std::string output = transform(source, solution);
  ASSERT_EQ(output, expected);
}

TEST(TFHECircuitParametrization, boostrap) {
  std::string source = R"(
  func.func @main(%arg0: !TFHE.glwe<sk?> {TFHE.OId = 0 : i32}) -> !TFHE.glwe<sk?> {
    %lut = arith.constant dense<-1152921504606846976> : tensor<42xi64>
    %0 = "TFHE.bootstrap_glwe"(%arg0, %lut) {TFHE.OId = 1 : i32, key = #TFHE.bsk<sk?, sk?, -1, -1, -1, -1>} : (!TFHE.glwe<sk?>, tensor<42xi64>) -> !TFHE.glwe<sk?>
    return %0 : !TFHE.glwe<sk?>
  }
)";
  std::string expected = R"(module {
  func.func @main(%arg0: !TFHE.glwe<sk<0,1,1701>>) -> !TFHE.glwe<sk<1,1,1024>> {
    %cst = arith.constant dense<-1152921504606846976> : tensor<1024xi64>
    %0 = "TFHE.bootstrap_glwe"(%arg0, %cst) {key = #TFHE.bsk<sk<0,1,1701>, sk<1,1,1024>, 1024, 1, 2, 12>} : (!TFHE.glwe<sk<0,1,1701>>, tensor<1024xi64>) -> !TFHE.glwe<sk<1,1,1024>>
    return %0 : !TFHE.glwe<sk<1,1,1024>>
  }
}
)";
  concrete_optimizer::dag::CircuitSolution solution;
  // Add a first secret key
  auto sk0 = addSecretKey(solution, 3, 567);
  auto sk1 = addSecretKey(solution, 1, 1024);
  auto bsk = addBootstrapKey(solution, sk0, sk1, 2, 12);
  // %arg0
  addInstructionKey(solution, sk0, sk0);
  // %0
  addInstructionKey(solution, sk0, sk1, -1, bsk);
  std::string output = transform(source, solution);
  ASSERT_EQ(output, expected);
}

// Test the extra conversion keys used to switch between two partitions without
// boostrap
// TODO: Will be a fastKS
TEST(TFHECircuitParametrization, extra_conversion_key) {
  std::string source = R"(
  func.func @main(%arg0: !TFHE.glwe<sk?> {TFHE.OId = 0 : i32}, %arg1: !TFHE.glwe<sk?> {TFHE.OId = 1 : i32}, %arg2: i64) -> !TFHE.glwe<sk?> {
    // Partition 0
    %0 = "TFHE.add_glwe_int"(%arg0, %arg2) {TFHE.OId = 2 : i32} : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
    // Partition 1
    %1 = "TFHE.add_glwe"(%0, %arg1) {TFHE.OId = 3 : i32} : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
    return %1 : !TFHE.glwe<sk?>
  }
)";
  std::string expected = R"(module {
  func.func @main(%arg0: !TFHE.glwe<sk<0,1,6144>>, %arg1: !TFHE.glwe<sk<1,1,1024>>, %arg2: i64) -> !TFHE.glwe<sk<1,1,1024>> {
    %0 = "TFHE.add_glwe_int"(%arg0, %arg2) : (!TFHE.glwe<sk<0,1,6144>>, i64) -> !TFHE.glwe<sk<0,1,6144>>
    %1 = "TFHE.keyswitch_glwe"(%0) {key = #TFHE.ksk<sk<0,1,6144>, sk<1,1,1024>, 2, 12>} : (!TFHE.glwe<sk<0,1,6144>>) -> !TFHE.glwe<sk<1,1,1024>>
    %2 = "TFHE.add_glwe"(%1, %arg1) : (!TFHE.glwe<sk<1,1,1024>>, !TFHE.glwe<sk<1,1,1024>>) -> !TFHE.glwe<sk<1,1,1024>>
    return %2 : !TFHE.glwe<sk<1,1,1024>>
  }
}
)";
  concrete_optimizer::dag::CircuitSolution solution;
  // Add secret key for partition 0
  auto sk0 = addSecretKey(solution, 3, 2048);
  // Add secret key for partition 1
  auto sk1 = addSecretKey(solution, 1, 1024);
  // Extra conversion key
  auto ksk = addExtraKeyswitchKey(solution, sk0, sk1, 2, 12);
  std::vector<uint64_t> extra_conversion_keys{(uint64_t)ksk};
  // Add instruction keys
  // #0: %arg0 - partition 0
  addInstructionKey(solution, sk0, sk0);
  // #1: %arg1 - partition 1
  addInstructionKey(solution, sk1, sk1);
  // #2: %0 - partition 0 with conversion to partition 1
  addInstructionKey(solution, sk0, sk0, -1, -1, extra_conversion_keys);
  // #3: %1 - partition 1
  addInstructionKey(solution, sk1, sk1);
  std::string output = transform(source, solution);
  ASSERT_EQ(output, expected);
}
