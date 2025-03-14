// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETE_RUST_COMPILER_H
#define CONCRETE_RUST_COMPILER_H

#include "concrete-optimizer.hpp"
#include "concrete-protocol.capnp.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Keys.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/context.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/V0Parameters.h"
#include "cxx.h"
#include <_types/_uint64_t.h>
#include <_types/_uint8_t.h>
#include <cstddef>
#include <memory>
#include <sys/_types/_int64_t.h>
#include <vector>

using mlir::concretelang::CompilationContext;
using mlir::concretelang::CompilerEngine;

// Contains helpers not available from the original compiler api.
namespace concrete_rust {

struct CompilationOptions: mlir::concretelang::CompilationOptions {

  void set_display_optimizer_choice(bool val) {
    this->optimizerConfig.display = val;
  }

  void set_loop_parallelize(bool val) { this->loopParallelize = val; }

  void set_dataflow_parallelize(bool val) {
    this->dataflowParallelize = val;
  }

  void set_auto_parallelize(bool val) { this->autoParallelize = val; }

  void set_compress_evaluation_keys(bool val) {
    this->compressEvaluationKeys = val;
  }

  void set_compress_input_ciphertexts(bool val) {
    this->compressInputCiphertexts = val;
  }

  void set_p_error(double val) { this->optimizerConfig.p_error = val; }

  void set_global_p_error(double val) {
    this->optimizerConfig.global_p_error = val;
  }

  void set_optimizer_strategy(uint8_t val) {
    this->optimizerConfig.strategy =
        static_cast<mlir::concretelang::optimizer::Strategy>(val);
  }

  void set_optimizer_multi_parameter_strategy(uint8_t val) {
    this->optimizerConfig.multi_param_strategy =
        static_cast<concrete_optimizer::MultiParamStrategy>(val);
  }
  void set_enable_tlu_fusing(bool val) { this->enableTluFusing = val; }

  void set_print_tlu_fusing(bool val) { this->printTluFusing = val; }

  void set_enable_overflow_detection_in_simulation(bool val) {
    this->enableOverflowDetectionInSimulation = val;
  }
  void set_simulate(bool val) { this->simulate = val; }

  void set_composable(bool val) {
    this->optimizerConfig.composable = val;
  }

  void set_range_restriction(rust::Str json) {
    if (!json.empty()) {
      this->optimizerConfig.range_restriction =
          std::make_shared<concrete_optimizer::restriction::RangeRestriction>(
              concrete_optimizer::restriction::range_restriction_from_json(
                  json));
    }
  }

  void set_keyset_restriction(rust::Str json) {
    if (!json.empty()) {
      this->optimizerConfig.keyset_restriction =
          std::make_shared<concrete_optimizer::restriction::KeysetRestriction>(
              concrete_optimizer::restriction::keyset_restriction_from_json(
                  json));
    }
  }

  void set_security_level(uint64_t val) {
    this->optimizerConfig.security = val;
  }

  void add_composition_rule(rust::Str from_func, rust::usize from_pos,
                            rust::Str to_func, rust::usize to_pos) {
    this->optimizerConfig.composition_rules.push_back(
        mlir::concretelang::optimizer::CompositionRule{
            std::string(from_func), from_pos, std::string(to_func), to_pos});
  }
};

std::unique_ptr<CompilationOptions> _compilation_options_new() {
  return std::make_unique<CompilationOptions>(
      CompilationOptions{mlir::concretelang::CompilationOptions()});
}

struct Library: mlir::concretelang::Library {
  rust::String _get_program_info_json() const {
    return this->programInfo.value().writeJsonToString().value();
  }
  rust::String get_static_library_path() const {
    return this->getStaticLibraryPath();
  }
};

std::unique_ptr<Library> compile(rust::Str sources,
                                 const CompilationOptions &options,
                                 rust::Str outputDirPath) {
  auto cxxSources = (std::string)sources;
  auto cxxOutputDirPath = (std::string)outputDirPath;
  auto context = CompilationContext::createShared();
  auto engine = CompilerEngine(context);
  engine.setCompilationOptions(options);
  std::unique_ptr<llvm::MemoryBuffer> mb =
      llvm::MemoryBuffer::getMemBuffer(cxxSources);
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(mb), llvm::SMLoc());
  auto maybeLibrary =
      engine.compile(sm, cxxOutputDirPath, "", false, true, true, true);
  if (auto err = maybeLibrary.takeError()) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }
  mlir::concretelang::Library output = *maybeLibrary;
  return std::make_unique<Library>(Library{output});
}

template <typename T> struct Key: T {
  rust::Slice<const uint64_t> get_buffer() {
    auto buffer = this->getBuffer();
    return {buffer.data(), buffer.size()};
  }

  rust::String _get_info_json() const {
    return this->getInfo().writeJsonToString().value();
  }
};

typedef Key<concretelang::keys::LweSecretKey> LweSecretKey;
typedef Key<concretelang::keys::LweBootstrapKey> LweBootstrapKey;
typedef Key<concretelang::keys::LweKeyswitchKey> LweKeyswitchKey;
typedef Key<concretelang::keys::PackingKeyswitchKey> PackingKeyswitchKey;

typedef concretelang::csprng::SecretCSPRNG  SecretCsprng;

std::shared_ptr<SecretCsprng> _secret_csprng_new(uint64_t high, uint64_t low) {
  return std::make_shared<SecretCsprng>((static_cast<__uint128_t>(high) << 64) | low);
}

typedef concretelang::csprng::EncryptionCSPRNG  EncryptionCsprng;

std::shared_ptr<EncryptionCsprng> _encryption_csprng_new(uint64_t high, uint64_t low) {
    return std::make_shared<EncryptionCsprng>((static_cast<__uint128_t>(high) << 64) | low);
}
struct ServerKeyset: concretelang::keysets::ServerKeyset{
    size_t _lwe_bootstrap_keys_len() const {
        return this->lweBootstrapKeys.size();
    }

    const LweBootstrapKey &_lwe_bootstrap_keys_nth(size_t nth) const {
        return static_cast<const LweBootstrapKey &>(this->lweBootstrapKeys.at(nth));
    }

    size_t _lwe_keyswitch_keys_len() const {
        return this->lweKeyswitchKeys.size();
    }

    const LweKeyswitchKey &_lwe_keyswitch_keys_nth(size_t nth) const {
        return static_cast<const LweKeyswitchKey &>(this->lweKeyswitchKeys.at(nth));
    }

    size_t _packing_keyswitch_keys_len() const {
        return this->packingKeyswitchKeys.size();
    }

    const PackingKeyswitchKey &_packing_keyswitch_keys_nth(size_t nth) const {
        return static_cast<const PackingKeyswitchKey &>(this->packingKeyswitchKeys.at(nth));
    }
};

struct ClientKeyset: concretelang::keysets::ClientKeyset{
    size_t _lwe_secret_keys_len() const {
        return this->lweSecretKeys.size();
    }

    const LweSecretKey &_lwe_secret_keys_nth(size_t nth) const {
        return static_cast<const LweSecretKey &>(this->lweSecretKeys.at(nth));
    }
};

struct Keyset: concretelang::keysets::Keyset {

  std::unique_ptr<ServerKeyset> get_server() const {
      auto copy = ServerKeyset{this->server};
      return std::make_unique<ServerKeyset>(copy);
  }

  std::unique_ptr<ClientKeyset> get_client() const {
      auto copy = ClientKeyset{this->client};
      return std::make_unique<ClientKeyset>(copy);
  }
};

std::unique_ptr<Keyset> _keyset_new(rust::Str keyset_info,
                                   SecretCsprng &secret_csprng,
                                   EncryptionCsprng &encryption_csprng) {
  auto info = Message<concreteprotocol::KeysetInfo>();
  info.readJsonFromString({keyset_info}).value();
  auto inner = concretelang::keysets::Keyset(info, secret_csprng,
                                             encryption_csprng);
  Keyset keyset = {inner};
  return std::make_unique<Keyset>(keyset);
}

struct ClientCircuit: concretelang::clientlib::ClientCircuit {};

struct ClientProgram: concretelang::clientlib::ClientProgram {
    std::unique_ptr<ClientCircuit> _get_client_circuit(rust::Str name) const {
        auto client_circuit = this->getClientCircuit({name}).value();
        ClientCircuit output = {client_circuit};
        return std::make_unique<ClientCircuit>(output);
    }
};

std::unique_ptr<ClientProgram> _client_program_new_encrypted(rust::Str program_info_json, const ClientKeyset & client_keyset, std::shared_ptr<EncryptionCsprng> csprng){
    auto info = Message<concreteprotocol::ProgramInfo>();
    info.readJsonFromString({program_info_json}).value();
    auto inner = concretelang::clientlib::ClientProgram::createEncrypted(info, client_keyset, csprng).value();
    ClientProgram output = {inner};
    return std::make_unique<ClientProgram>(output);
}

std::unique_ptr<ClientProgram> _client_program_new_simulated(rust::Str program_info_json, std::shared_ptr<EncryptionCsprng> csprng){
    auto info = Message<concreteprotocol::ProgramInfo>();
    info.readJsonFromString({program_info_json}).value();
    auto inner = concretelang::clientlib::ClientProgram::createSimulated(info, csprng).value();
    ClientProgram output = {inner};
    return std::make_unique<ClientProgram>(output);
}

typedef Tensor<uint8_t> TensorU8;
typedef Tensor<int8_t> TensorI8;
typedef Tensor<uint16_t> TensorU16;
typedef Tensor<int16_t> TensorI16;
typedef Tensor<uint32_t> TensorU32;
typedef Tensor<int32_t> TensorI32;
typedef Tensor<uint64_t> TensorU64;
typedef Tensor<int64_t> TensorI64;

template<typename T, typename O>
std::unique_ptr<O> _tensor_new(rust::Slice<const T> values, rust::Slice<const size_t> dimensions) {
    const std::vector<T> values_vec = {values.begin(), values.end()};
    const std::vector<size_t> dimensions_vec = {dimensions.begin(), dimensions.end()};
    const O output = {values_vec, dimensions_vec};
    return std::make_unique<O>(output);
}

const auto _tensor_u8_new = _tensor_new<uint8_t, TensorU8>;
const auto _tensor_i8_new = _tensor_new<int8_t, TensorI8>;
const auto _tensor_u16_new = _tensor_new<uint16_t, TensorU16>;
const auto _tensor_i16_new = _tensor_new<int16_t, TensorI16>;
const auto _tensor_u32_new = _tensor_new<uint32_t, TensorU32>;
const auto _tensor_i32_new = _tensor_new<int32_t, TensorI32>;
const auto _tensor_u64_new = _tensor_new<uint64_t, TensorU64>;
const auto _tensor_i64_new = _tensor_new<int64_t, TensorI64>;



typedef concretelang::values::Value Value;
typedef concretelang::values::TransportValue TransportValue;

} // namespace concrete_rust

#endif
