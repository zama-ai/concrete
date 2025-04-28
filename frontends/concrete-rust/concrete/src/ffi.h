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
#include "concretelang/ServerLib/ServerLib.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/V0Parameters.h"
#include "cxx.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <sys/types.h>
#include <vector>

using mlir::concretelang::CompilationContext;
using mlir::concretelang::CompilerEngine;

inline constexpr capnp::ReaderOptions DESER_OPTIONS = {7000000000, 64};

class VecOStream : public std::streambuf {
public:
    VecOStream(rust::Vec<uint8_t>& vec) : vec_(vec) {}
protected:
    int overflow(int c) override {
        if (c != EOF) {
            vec_.push_back(static_cast<uint8_t>(c));
        }
        return c;
    }
private:
    rust::Vec<uint8_t>& vec_;
};

class SliceIStream : public std::streambuf {
public:
    SliceIStream(const rust::Slice<const uint8_t>& slice) : slice_(slice), pos_(0) {
        setg(reinterpret_cast<char*>(const_cast<uint8_t*>(slice_.data())),
             reinterpret_cast<char*>(const_cast<uint8_t*>(slice_.data())),
             reinterpret_cast<char*>(const_cast<uint8_t*>(slice_.data() + slice_.size())));
    }
protected:
    int underflow() override {
        if (pos_ >= slice_.size()) {
            return EOF;
        }
        return slice_[pos_++];
    }
private:
    const rust::Slice<const uint8_t>& slice_;
    size_t pos_;
};

namespace concrete_rust {

struct CompilationOptions : mlir::concretelang::CompilationOptions {

  void set_display_optimizer_choice(bool val) {
    this->optimizerConfig.display = val;
  }

  void set_loop_parallelize(bool val) { this->loopParallelize = val; }

  void set_dataflow_parallelize(bool val) { this->dataflowParallelize = val; }

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

  void set_composable(bool val) { this->optimizerConfig.composable = val; }

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
  auto output = std::make_unique<mlir::concretelang::CompilationOptions>();
  return std::unique_ptr<CompilationOptions>(
      reinterpret_cast<CompilationOptions *>(output.release()));
}

struct Library : mlir::concretelang::Library {
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
  return std::make_unique<Library>(Library{*maybeLibrary});
}

template <typename T> struct Key : T {

  static std::unique_ptr<Key<T>> _from_buffer_and_info(rust::Slice<const uint64_t> buffer_slice, rust::Str info_json) {
      auto info = typename T::InfoType();
      auto info_string = std::string(info_json);
      assert(info.readJsonFromString(info_string).has_value());
      auto buffer = std::make_shared<std::vector<uint64_t>>(buffer_slice.begin(), buffer_slice.end());
      auto output =std::make_unique<T>(buffer, info);
      return std::unique_ptr<Key<T>>(reinterpret_cast<Key<T> *>(output.release()));
  }

  rust::Slice<const uint64_t> get_buffer() {
    auto buffer = this->getBuffer();
    return {buffer.data(), buffer.size()};
  }
  rust::String _get_info_json() const {
    return this->getInfo().writeJsonToString().value();
  }
};

typedef Key<concretelang::keys::LweSecretKey> LweSecretKey;

inline std::unique_ptr<LweSecretKey>
_lwe_secret_key_from_buffer_and_info(rust::Slice<const uint64_t> buffer_slice, rust::Str info_json) {
    return LweSecretKey::_from_buffer_and_info(buffer_slice, info_json);
}

typedef Key<concretelang::keys::LweBootstrapKey> LweBootstrapKey;
typedef Key<concretelang::keys::LweKeyswitchKey> LweKeyswitchKey;
typedef Key<concretelang::keys::PackingKeyswitchKey> PackingKeyswitchKey;

typedef concretelang::csprng::SecretCSPRNG SecretCsprng;

std::unique_ptr<SecretCsprng> _secret_csprng_new(uint64_t high, uint64_t low) {
  return std::make_unique<SecretCsprng>((static_cast<__uint128_t>(high) << 64) |
                                        low);
}

typedef concretelang::csprng::EncryptionCSPRNG EncryptionCsprng;

std::unique_ptr<EncryptionCsprng> _encryption_csprng_new(uint64_t high,
                                                         uint64_t low) {
  return std::make_unique<EncryptionCsprng>(
      (static_cast<__uint128_t>(high) << 64) | low);
}

struct ServerKeyset : concretelang::keysets::ServerKeyset {
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
    return static_cast<const PackingKeyswitchKey &>(
        this->packingKeyswitchKeys.at(nth));
  }

  rust::Vec<uint8_t> serialize() const {
      auto proto = toProto();
      auto output = rust::Vec<uint8_t>();
      auto vec_ostream = VecOStream(output);
      auto ostream = std::ostream(&vec_ostream);
      proto.writeBinaryToOstream(
          ostream
      ).value();
      ostream.flush();
      return output;
  }
};

std::unique_ptr<ServerKeyset> _deserialize_server_keyset(rust::Slice<const uint8_t> slice) {
    auto proto = Message<concreteprotocol::ServerKeyset>();
    auto slice_istream = SliceIStream(slice);
    auto istream = std::istream(&slice_istream);
    proto.readBinaryFromIstream(istream, DESER_OPTIONS).value();
    auto output = std::make_unique<concretelang::keysets::ServerKeyset>(ServerKeyset::fromProto(proto));
    return std::unique_ptr<ServerKeyset>(reinterpret_cast<ServerKeyset *>(output.release()));
}

struct ClientKeyset : concretelang::keysets::ClientKeyset {
  size_t _lwe_secret_keys_len() const { return this->lweSecretKeys.size(); }

  const LweSecretKey &_lwe_secret_keys_nth(size_t nth) const {
    return static_cast<const LweSecretKey &>(this->lweSecretKeys.at(nth));
  }

  rust::Vec<uint8_t> serialize() const {
      auto proto = toProto();
      auto output = rust::Vec<uint8_t>();
      auto vec_ostream = VecOStream(output);
      auto ostream = std::ostream(&vec_ostream);
      proto.writeBinaryToOstream(
          ostream
      ).value();
      ostream.flush();
      return output;
  }
};

std::unique_ptr<ClientKeyset> _deserialize_client_keyset(rust::Slice<const uint8_t> slice) {
    auto proto = Message<concreteprotocol::ClientKeyset>();
    auto slice_istream = SliceIStream(slice);
    auto istream = std::istream(&slice_istream);
    proto.readBinaryFromIstream(istream, DESER_OPTIONS).value();
    auto output = std::make_unique<concretelang::keysets::ClientKeyset>(ClientKeyset::fromProto(proto));
    return std::unique_ptr<ClientKeyset>(reinterpret_cast<ClientKeyset *>(output.release()));
}

struct Keyset : concretelang::keysets::Keyset {

  std::unique_ptr<ServerKeyset> get_server() const {
    auto copy =
        std::make_unique<concretelang::keysets::ServerKeyset>(this->server);
    return std::unique_ptr<ServerKeyset>(
        reinterpret_cast<ServerKeyset *>(copy.release()));
  }

  std::unique_ptr<ClientKeyset> get_client() const {
    auto copy =
        std::make_unique<concretelang::keysets::ClientKeyset>(this->client);
    return std::unique_ptr<ClientKeyset>(
        reinterpret_cast<ClientKeyset *>(copy.release()));
  }
};

std::unique_ptr<Keyset> _keyset_new(rust::Str keyset_info,
                                    SecretCsprng &secret_csprng,
                                    EncryptionCsprng &encryption_csprng,
                                    rust::Slice<std::unique_ptr<LweSecretKey>> initial_keys) {
  auto info = Message<concreteprotocol::KeysetInfo>();
  info.readJsonFromString(std::string(keyset_info)).value();
  auto map = std::map<uint32_t, concretelang::keys::LweSecretKey>();
  for (auto &key : initial_keys) {
      auto info = key->getInfo();
      map.insert(std::make_pair(info.asReader().getId(), std::move(*key.release())));
  }
  auto output = std::make_unique<concretelang::keysets::Keyset>(
      info, secret_csprng, encryption_csprng, map);
  return std::unique_ptr<Keyset>(reinterpret_cast<Keyset *>(output.release()));
}

template <typename T> struct Tensor : ::concretelang::values::Tensor<T> {
  rust::Slice<const size_t> _get_dimensions() const {
    return {this->dimensions.data(), this->dimensions.size()};
  }
  rust::Slice<const T> _get_values() const {
    return {this->values.data(), this->values.size()};
  }
};

typedef Tensor<uint8_t> TensorU8;
typedef Tensor<int8_t> TensorI8;
typedef Tensor<uint16_t> TensorU16;
typedef Tensor<int16_t> TensorI16;
typedef Tensor<uint32_t> TensorU32;
typedef Tensor<int32_t> TensorI32;
typedef Tensor<uint64_t> TensorU64;
typedef Tensor<int64_t> TensorI64;

template <typename T, typename O>
std::unique_ptr<O> _tensor_new(rust::Slice<const T> values,
                               rust::Slice<const size_t> dimensions) {
  std::vector<T> values_vec = {values.begin(), values.end()};
  std::vector<size_t> dimensions_vec = {dimensions.begin(),
                                              dimensions.end()};
  auto output = std::make_unique<::concretelang::values::Tensor<T>>(values_vec, dimensions_vec);
  return std::unique_ptr<O>(reinterpret_cast<O *>(output.release()));
}
const auto _tensor_u8_new = _tensor_new<uint8_t, TensorU8>;
const auto _tensor_i8_new = _tensor_new<int8_t, TensorI8>;
const auto _tensor_u16_new = _tensor_new<uint16_t, TensorU16>;
const auto _tensor_i16_new = _tensor_new<int16_t, TensorI16>;
const auto _tensor_u32_new = _tensor_new<uint32_t, TensorU32>;
const auto _tensor_i32_new = _tensor_new<int32_t, TensorI32>;
const auto _tensor_u64_new = _tensor_new<uint64_t, TensorU64>;
const auto _tensor_i64_new = _tensor_new<int64_t, TensorI64>;

struct TransportValue : concretelang::values::TransportValue {
  std::unique_ptr<TransportValue> to_owned() const {
    return std::make_unique<TransportValue>(*this);
  }

  rust::Vec<uint8_t> serialize() const {
      auto output = rust::Vec<uint8_t>();
      auto vec_ostream = VecOStream(output);
      auto ostream = std::ostream(&vec_ostream);
      this->writeBinaryToOstream(
          ostream
      ).value();
      ostream.flush();
      return output;
  }

};

struct Value : concretelang::values::Value {
  bool _has_element_type_u8() const { return hasElementType<uint8_t>(); }
  bool _has_element_type_i8() const { return hasElementType<int8_t>(); }
  bool _has_element_type_u16() const { return hasElementType<uint16_t>(); }
  bool _has_element_type_i16() const { return hasElementType<int16_t>(); }
  bool _has_element_type_u32() const { return hasElementType<uint32_t>(); }
  bool _has_element_type_i32() const { return hasElementType<int32_t>(); }
  bool _has_element_type_u64() const { return hasElementType<uint64_t>(); }
  bool _has_element_type_i64() const { return hasElementType<int64_t>(); }
  std::unique_ptr<TensorI8> _get_tensor_i8() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<int8_t>>(std::move(getTensor<int8_t>().value()));
    return std::unique_ptr<TensorI8>(reinterpret_cast<TensorI8 *>(output.release()));
  }

  std::unique_ptr<TensorU8> _get_tensor_u8() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<uint8_t>>(std::move(getTensor<uint8_t>().value()));
    return std::unique_ptr<TensorU8>(reinterpret_cast<TensorU8 *>(output.release()));
  }

  std::unique_ptr<TensorI16> _get_tensor_i16() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<int16_t>>(std::move(getTensor<int16_t>().value()));
    return std::unique_ptr<TensorI16>(reinterpret_cast<TensorI16 *>(output.release()));
  }

  std::unique_ptr<TensorU16> _get_tensor_u16() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<uint16_t>>(std::move(getTensor<uint16_t>().value()));
    return std::unique_ptr<TensorU16>(reinterpret_cast<TensorU16 *>(output.release()));
  }

  std::unique_ptr<TensorI32> _get_tensor_i32() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<int32_t>>(std::move(getTensor<int32_t>().value()));
    return std::unique_ptr<TensorI32>(reinterpret_cast<TensorI32 *>(output.release()));
  }

  std::unique_ptr<TensorU32> _get_tensor_u32() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<uint32_t>>(std::move(getTensor<uint32_t>().value()));
    return std::unique_ptr<TensorU32>(reinterpret_cast<TensorU32 *>(output.release()));
  }

  std::unique_ptr<TensorI64> _get_tensor_i64() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<int64_t>>(std::move(getTensor<int64_t>().value()));
    return std::unique_ptr<TensorI64>(reinterpret_cast<TensorI64 *>(output.release()));
  }

  std::unique_ptr<TensorU64> _get_tensor_u64() const {
    auto output =
        std::make_unique<::concretelang::values::Tensor<uint64_t>>(std::move(getTensor<uint64_t>().value()));
    return std::unique_ptr<TensorU64>(reinterpret_cast<TensorU64 *>(output.release()));
  }

  rust::Slice<const size_t> get_dimensions() const {
    const auto& vecref = getDimensions();
    return {vecref.data(), vecref.size()};
  }

  std::unique_ptr<TransportValue> into_transport_value(rust::Str type_info_json) const {
      auto first = intoRawTransportValue();
      auto info = Message<concreteprotocol::TypeInfo>();
      info.readJsonFromString(std::string(type_info_json)).value();
      first.asBuilder().setTypeInfo(info.asReader());
      auto output =
          std::make_unique<::concretelang::values::TransportValue>(first);
      return std::unique_ptr<TransportValue>(reinterpret_cast<TransportValue *>(output.release()));
  }
};

template <typename T>
std::unique_ptr<Value> _value_from_tensor(std::unique_ptr<T> tensor) {
  T otensor = *tensor;
  auto val = std::make_unique<concretelang::values::Value>(otensor);
  return std::unique_ptr<Value>(reinterpret_cast<Value *>(val.release()));
}
const auto _value_from_tensor_u8 = _value_from_tensor<TensorU8>;
const auto _value_from_tensor_i8 = _value_from_tensor<TensorI8>;
const auto _value_from_tensor_u16 = _value_from_tensor<TensorU16>;
const auto _value_from_tensor_i16 = _value_from_tensor<TensorI16>;
const auto _value_from_tensor_u32 = _value_from_tensor<TensorU32>;
const auto _value_from_tensor_i32 = _value_from_tensor<TensorI32>;
const auto _value_from_tensor_u64 = _value_from_tensor<TensorU64>;
const auto _value_from_tensor_i64 = _value_from_tensor<TensorI64>;

std::unique_ptr<TransportValue> _deserialize_transport_value(rust::Slice<const uint8_t> slice) {
    auto output = TransportValue();
    auto slice_istream = SliceIStream(slice);
    auto istream = std::istream(&slice_istream);
    output.readBinaryFromIstream(istream).value();
    return std::make_unique<TransportValue>(output);
}

std::unique_ptr<Value> _transport_value_to_value(TransportValue const &tv) {
    auto output =
        std::make_unique<::concretelang::values::Value>(::concretelang::values::Value::fromRawTransportValue(tv));
    return std::unique_ptr<Value>(reinterpret_cast<Value *>(output.release()));
}

struct ClientFunction : concretelang::clientlib::ClientCircuit {
  std::unique_ptr<TransportValue> prepare_input(std::unique_ptr<Value> arg,
                                                size_t pos) {
    auto oarg = *arg.release();
    auto output = std::make_unique<::concretelang::values::TransportValue>(prepareInput(oarg, pos).value());
    return std::unique_ptr<TransportValue>(
        reinterpret_cast<TransportValue *>(output.release()));
  }
  std::unique_ptr<TransportValue> simulate_prepare_input(const Value &arg,
                                                         size_t pos) {
    Value oarg = {arg};
    auto output = std::make_unique<::concretelang::values::TransportValue>(simulatePrepareInput(oarg, pos).value());
    return std::unique_ptr<TransportValue>(
        reinterpret_cast<TransportValue *>(output.release()));
  }
  std::unique_ptr<Value> process_output(std::unique_ptr<TransportValue> result,
                                        size_t pos) {
    auto oresult = *result.release();
    auto output = std::make_unique<::concretelang::values::Value>(processOutput(oresult, pos).value());
    return std::unique_ptr<Value>(reinterpret_cast<Value *>(output.release()));
  }
  std::unique_ptr<Value> simulate_process_output(const TransportValue &result,
                                                 size_t pos) {
    TransportValue oresult = {result};
    auto output = std::make_unique<::concretelang::values::Value>(simulateProcessOutput(oresult, pos).value());
    return std::unique_ptr<Value>(reinterpret_cast<Value *>(output.release()));
  }
};

std::unique_ptr<ClientFunction>
_client_function_new_encrypted(rust::Str circuit_info_json,
                              const ClientKeyset &client_keyset,
                              std::unique_ptr<EncryptionCsprng> csprng) {
  auto info = Message<concreteprotocol::CircuitInfo>();
  info.readJsonFromString(std::string(circuit_info_json)).value();
  auto inner = std::make_unique<::concretelang::clientlib::ClientCircuit>(::concretelang::clientlib::ClientCircuit::createEncrypted(
                   info, client_keyset, std::move(csprng))
                   .value());
  return std::unique_ptr<ClientFunction>(
      reinterpret_cast<ClientFunction *>(inner.release()));
}

std::unique_ptr<ClientFunction>
_client_function_new_simulated(rust::Str circuit_info_json,
                              std::unique_ptr<EncryptionCsprng> csprng) {
  auto info = Message<concreteprotocol::CircuitInfo>();
  info.readJsonFromString(std::string(circuit_info_json)).value();
  auto inner = std::make_unique<::concretelang::clientlib::ClientCircuit>(::concretelang::clientlib::ClientCircuit::createSimulated(
                   info, std::move(csprng))
                   .value());
  return std::unique_ptr<ClientFunction>(
      reinterpret_cast<ClientFunction *>(inner.release()));
}

struct ClientModule : concretelang::clientlib::ClientProgram {
  std::unique_ptr<ClientFunction> _get_client_function(rust::Str name) const {
    auto output = std::make_unique<::concretelang::clientlib::ClientCircuit>(this->getClientCircuit(std::string(name)).value());
    return std::unique_ptr<ClientFunction>(
        reinterpret_cast<ClientFunction *>(output.release()));
  }
};

std::unique_ptr<ClientModule>
_client_module_new_encrypted(rust::Str program_info_json,
                              const ClientKeyset &client_keyset,
                              std::unique_ptr<EncryptionCsprng> csprng) {
  auto info = Message<concreteprotocol::ProgramInfo>();
  info.readJsonFromString(std::string(program_info_json)).value();
  auto output = std::make_unique<::concretelang::clientlib::ClientProgram>(::concretelang::clientlib::ClientProgram::createEncrypted(
                    info, client_keyset, std::move(csprng))
                    .value());
  return std::unique_ptr<ClientModule>(
      reinterpret_cast<ClientModule *>(output.release()));
}

std::unique_ptr<ClientModule>
_client_module_new_simulated(rust::Str program_info_json,
                              std::unique_ptr<EncryptionCsprng> csprng) {
  auto info = Message<concreteprotocol::ProgramInfo>();
  info.readJsonFromString(std::string(program_info_json)).value();
  auto output = std::make_unique<::concretelang::clientlib::ClientProgram>(::concretelang::clientlib::ClientProgram::createSimulated(
                    info, std::move(csprng))
                    .value());
  return std::unique_ptr<ClientModule>(
      reinterpret_cast<ClientModule *>(output.release()));
}

struct ServerFunction : concretelang::serverlib::ServerCircuit {
  std::unique_ptr<std::vector<TransportValue>>
  _call(const ServerKeyset &keys,
        rust::Slice<std::unique_ptr<TransportValue>> args) {
    std::vector<concretelang::values::TransportValue> oargs{};
    for (size_t i = 0; i < args.length(); i++) {
      oargs.push_back(*args[i].release());
    }
    auto maybe_res = call(keys, oargs);
    if (maybe_res.has_error()){
        std::cout << "Failed to perform call:\n";
        std::cout << maybe_res.error().mesg;
        std::cout.flush();
        assert(false);
    }
    auto res = std::make_unique<std::vector<::concretelang::values::TransportValue>>(maybe_res.value());
    return std::unique_ptr<std::vector<TransportValue>>(
        reinterpret_cast<std::vector<TransportValue> *>(res.release()));
  }

  std::unique_ptr<std::vector<TransportValue>>
  _simulate(rust::Slice<std::unique_ptr<TransportValue>> args) {
    std::vector<concretelang::values::TransportValue> oargs{};
    for (size_t i = 0; i < args.length(); i++) {
      oargs.push_back(*args[i].release());
    }
    auto res = std::make_unique<std::vector<::concretelang::values::TransportValue>>(simulate(oargs).value());
    return std::unique_ptr<std::vector<TransportValue>>(
        reinterpret_cast<std::vector<TransportValue> *>(res.release()));
  }
};

typedef void (*FnPtr)(void *, ...);
using c_void = void;

std::unique_ptr<ServerFunction> _server_function_new(rust::Str circuit_info_json,
                                                   void *func,
                                                   bool use_simulation) {
  auto info = Message<concreteprotocol::CircuitInfo>();
  info.readJsonFromString(std::string(circuit_info_json)).value();
  FnPtr fn_ptr = reinterpret_cast<FnPtr>(func);
  auto output = concretelang::serverlib::ServerCircuit::fromFnPtr(
                    info, fn_ptr, use_simulation)
                    .value();
  return std::make_unique<ServerFunction>(
      *reinterpret_cast<ServerFunction *>(&output));
}

} // namespace concrete_rust

#endif
