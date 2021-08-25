#ifndef ZAMALANG_SUPPORT_COMPILERTOOLS_H_
#define ZAMALANG_SUPPORT_COMPILERTOOLS_H_

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Pass/PassManager.h>

#include "zamalang/Support/ClientParameters.h"
#include "zamalang/Support/KeySet.h"
#include "zamalang/Support/V0Parameters.h"

namespace mlir {
namespace zamalang {

class CompilerTools {
public:
  struct LowerOptions {
    llvm::function_ref<bool(std::string)> enablePass;
    bool verbose;

    LowerOptions()
        : verbose(false), enablePass([](std::string pass) { return true; }){};
  };

  /// lowerHLFHEToMlirLLVMDialect run all passes to lower FHE dialects to mlir
  /// lowerable to llvm dialect.
  /// The given module MLIR operation would be modified and the constraints set.
  static mlir::LogicalResult
  lowerHLFHEToMlirStdsDialect(mlir::MLIRContext &context,
                              mlir::Operation *module, V0FHEContext &fheContext,
                              LowerOptions options = LowerOptions());

  /// lowerMlirStdsDialectToMlirLLVMDialect run all passes to lower MLIR
  /// dialects to MLIR LLVM dialect. The given module MLIR operation would be
  /// modified.
  static mlir::LogicalResult
  lowerMlirStdsDialectToMlirLLVMDialect(mlir::MLIRContext &context,
                                        mlir::Operation *module,
                                        LowerOptions options = LowerOptions());

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  toLLVMModule(llvm::LLVMContext &llvmContext, mlir::ModuleOp &module,
               llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline);
};

/// JITLambda is a tool to JIT compile an mlir module and to invoke a function
/// of the module.
class JITLambda {
public:
  class Argument {
  public:
    Argument(KeySet &keySet);
    ~Argument();

    // Create lambda Argument that use the given KeySet to perform encryption
    // and decryption operations.
    static llvm::Expected<std::unique_ptr<Argument>> create(KeySet &keySet);

    // Set a scalar argument at the given pos as a uint64_t.
    llvm::Error setArg(size_t pos, uint64_t arg);

    // Set a argument at the given pos as a tensor of int64.
    llvm::Error setArg(size_t pos, uint64_t *data, size_t size) {
      return setArg(pos, 64, (void *)data, size);
    }

    // Set a argument at the given pos as a tensor of int32.
    llvm::Error setArg(size_t pos, uint32_t *data, size_t size) {
      return setArg(pos, 32, (void *)data, size);
    }

    // Set a argument at the given pos as a tensor of int32.
    llvm::Error setArg(size_t pos, uint16_t *data, size_t size) {
      return setArg(pos, 16, (void *)data, size);
    }

    // Set a tensor argument at the given pos as a uint64_t.
    llvm::Error setArg(size_t pos, uint8_t *data, size_t size) {
      return setArg(pos, 8, (void *)data, size);
    }

    // Get the result at the given pos as an uint64_t.
    llvm::Error getResult(size_t pos, uint64_t &res);

    // Fill the result.
    llvm::Error getResult(size_t pos, uint64_t *res, size_t size);

  private:
    llvm::Error setArg(size_t pos, size_t width, void *data, size_t size);

    friend JITLambda;
    // Store the pointer on inputs values and outputs values
    std::vector<void *> rawArg;
    // Store the values of inputs
    std::vector<void *> inputs;
    // Store the values of outputs
    std::vector<void *> outputs;
    // Store the input gates description and the offset of the argument.
    std::vector<std::tuple<CircuitGate, size_t /*offet*/>> inputGates;
    // Store the outputs gates description and the offset of the argument.
    std::vector<std::tuple<CircuitGate, size_t /*offet*/>> outputGates;
    // Store allocated lwe ciphertexts (for free)
    std::vector<LweCiphertext_u64 *> allocatedCiphertexts;
    // Store buffers of ciphertexts
    std::vector<LweCiphertext_u64 **> ciphertextBuffers;

    KeySet &keySet;
  };
  JITLambda(mlir::LLVM::LLVMFunctionType type, llvm::StringRef name)
      : type(type), name(name){};

  /// create a JITLambda that point to the function name of the given module.
  static llvm::Expected<std::unique_ptr<JITLambda>>
  create(llvm::StringRef name, mlir::ModuleOp &module,
         llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline);

  /// invokeRaw execute the jit lambda with a list of Argument, the last one is
  /// used to store the result of the computation.
  /// Example:
  /// uin64_t arg0 = 1;
  /// uin64_t res;
  /// llvm::SmallVector<void *> args{&arg1, &res};
  /// lambda.invokeRaw(args);
  llvm::Error invokeRaw(llvm::MutableArrayRef<void *> args);

  /// invoke the jit lambda with the Argument.
  llvm::Error invoke(Argument &args);

private:
  mlir::LLVM::LLVMFunctionType type;
  llvm::StringRef name;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

} // namespace zamalang
} // namespace mlir

#endif