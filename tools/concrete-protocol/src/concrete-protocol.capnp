# Concrete Protocol
# 
# The following document contains a programatic description of a communication protocol to store and 
# exchange data with applications of the concrete framework. 
#
# Todo:
#   + Use `storagePrecision` instead of `integerPrecision` to better differentiate between the 
#     message and the storage.
#   + Use `storageInfo` instead of `rawInfo`.

@0xd2a64233258d00f1;

using Cxx = import "/capnp/c++.capnp";

$Cxx.namespace("concreteprotocol");

######################################################################################### Commons ##

enum KeyType {
  # Secret Keys can be drawn from different ranges of values, using different distributions. This 
  # enumeration encodes the different supported ways.
  
  binary @0; # Uniform sampling in {0, 1}
  ternary @1; # Uniform sampling in {-1, 0, 1}
}

struct Modulus {
  # Ciphertext operations are performed using modular arithmetic. Depending on the use, different 
  # modulus can be used for the operations. This structure encodes the different supported ways. 

  modulus :union $Cxx.name("mod") {
    # The modulus expected to be used.
    
    native @0 :NativeModulus;
    powerOfTwo @1 :PowerOfTwoModulus;
    integer @2 :IntegerModulus;
  }
}

struct NativeModulus{
  # Operations are performed using the modulus of the integers used to store the ciphertexts. 
  # 
  # Note:
  #   The bitwidth of the integer storage is represented implicitly here, and must be grabbed from 
  #   the rest of the description.
  # 
  # Example:
  #   2^64 when the ciphertext is stored using 64 bits integers.
}

struct PowerOfTwoModulus{
  # Operations are performed using a modulus that is a power of two. 
  # 
  # Example: 
  #   2^n for any n between 0 and the bitwidth of the integer used to store the ciphertext.
       
  power @0 :UInt32; # The power used to raise 2.
}

struct IntegerModulus{
  # Operations are performed using a modulus that is an arbitrary integer. 
  #
  # Example:
  #   n for any n between 0 and 2^N where N is the bitwidth of the integer used to store the 
  #   ciphertext.
    
  modulus @0 :UInt32; # The value used as modulus.
}

struct Shape{
  # Scalar and tensor values are represented by the same types. This structure contains a 
  # description of the shape of value. 
  #
  # Note:
  #   If the dimensions vector is empty, the message is interpreted as a scalar.
       
  dimensions @0 :List(UInt32); # The dimensions of the value.
}

struct RawInfo{
  # A value exchanged at the boundary between two parties of a computation will be transmitted as a 
  # binary payload containing a tensor of integers. This payload will first have to be parsed to a 
  # tensor of proper shape, signedness and precision before being pre-processed and passed to the 
  # computation. This structure represents the informations needed to parse this payload into the 
  # expected tensor.

  shape @0 :Shape; # The shape of the tensor.
  integerPrecision @1 :UInt32; # The precision of the integers.
  isSigned @2 :Bool; # The signedness of the integers.
}

struct Payload{
  # A structure carrying a binary payload.
  #
  # Note:
  #   There is a limit to the maximum size of a Data type. For this reason, large payloads must be 
  #   split into several blobs stored sequentially in a list. All but the last blobs store the 
  #   maximum amount of data allowed by Data, and the last store the remainder.
  data @0 :List(Data); # The binary data of the payload
}

##################################################################################### Compression ##

enum Compression{
  # Evaluation keys and ciphertexts can be compressed when transported over the wire. This 
  # enumeration encodes the different compressions that can be used to compress scheme objects.
  # 
  # Note:
  #   Not all compressions are available for every types of evaluation keys or ciphertexts.
  
  none @0; # No compression is used.
  seed @1; # The mask is represented by the seed of a csprng.
  paillier @2; # An output lwe ciphertext transciphered to the paillier cryptosystem.
}

################################################################################# LWE secret keys ##

struct LweSecretKeyParams {
  # A secret key is parameterized by a few quantities of cryptographic importance. This structure 
  # represents those parameters.
    
  lweDimension @0 :UInt32; # The LWE dimension, e.g. the length of the key.
  integerPrecision @1 :UInt32; # The bitwidth of the integers used for storage.
  keyType @2 :KeyType; # The kind of distribution used to sample the key.
}

struct LweSecretKeyInfo {
  # A secret key value is uniquely described by cryptographic parameters and an identifier. This 
  # structure represents this description of a secret key.
  # 
  # Note:
  #   Secret keys with same parameters are allowed to co-exist in a program, as long as they 
  #   have different ids.
    
  id @0 :UInt32; # The identifier of the key.
  params @1 :LweSecretKeyParams; # The cryptographic parameters of the keys.
}

struct LweSecretKey {
  # A secret key value is a payload and a description to interpret this payload. This structure
  # can be used to store and communicate a secret key.
  
  info @0 :LweSecretKeyInfo; # The description of the secret key.
  payload @1 :Payload; # The payload
}

################################################################################# LWE public keys ##

struct LwePublicKeyParams {
  # A public key is parameterized by a few quantities of cryptographic importance. This structure
  # represents those parameters.

  lweDimension @0 :UInt32; # The LWE dimension, e.g. the length of the corresponding secret key.
  integerPrecision @1 :UInt32; # The bitwidth of the integers used for storage.
  zeroEncryptionCount @2 :UInt32; # The number of LWE encryptions of 0 in a public key.
  variance @3 :Float64; # The variance used to encrypt the ciphertexts.
}

struct LweCompactPublicKeyParams {
  # A public key is parameterized by a few quantities of cryptographic importance. This structure
  # represents those parameters.

  lweDimension @0 :UInt32; # The LWE dimension, e.g. the length of the corresponding secret key.
  integerPrecision @1 :UInt32; # The bitwidth of the integers used for storage.
  variance @2 :Float64; # The variance used to encrypt the ciphertexts.
}

struct LwePublicKeyInfo {
  # A public key value is uniquely described by cryptographic parameters and an identifier. This
  # structure represents this description of a public key.

  id @0 :UInt32; # The identifier of the key.

  params :union {
    # The cryptographic parameters of the keys.

    classic @1 :LwePublicKeyParams;
    compact @2 :LweCompactPublicKeyParams;
  }
}

struct LwePublicKey {
  # A public key value is a payload and a description to interpret this payload. This structure
  # can be used to store and communicate a public key.

  info @0 :LwePublicKeyInfo; # The description of the public key.
  payload @1 :Payload; # The payload
}

############################################################################## LWE bootstrap keys ##

struct LweBootstrapKeyParams {
  # A bootstrap key is parameterized by a few quantities of cryptographic importance. This structure 
  # represents those parameters.
  #
  # Note:
  #   For now, only keys with the same input and output key types can be represented. 
    
  levelCount @0 :UInt32; # The number of levels of the ciphertexts.
  baseLog @1 :UInt32; # The logarithm of the base of the ciphertext.
  glweDimension @2 :UInt32; # The dimension of the ciphertexts.
  polynomialSize @3 :UInt32; # The polynomial size of the ciphertexts.
  inputLweDimension @8 :UInt32; # The dimension of the input lwe secret key.
  variance @4 :Float64; # The variance used to encrypt the ciphertexts.
  integerPrecision @5 :UInt32; # The bitwidth of the integers used to store the ciphertexts.
  modulus @6 :Modulus; # The modulus used to perform operations with this key.
  keyType @7 :KeyType; # The distribution of the input and output secret keys.
}

struct LweBootstrapKeyInfo {
  # A bootstrap key value is uniquely described by cryptographic parameters and a few application
  # related quantities. This structure represents this description of a bootstrap key.
  # 
  # Note:
  #   Bootstrap keys with same parameters, compression, input and output id, are allowed to co-exist
  #   in a program as long as they have different ids.

  id @0 :UInt32; # The identifier of the bootstrap key.
  inputId @1 :UInt32; # The identifier of the input secret key.
  outputId @2 :UInt32; # The identifier of the output secret key.
  params @3 :LweBootstrapKeyParams; # The cryptographic parameters of the key.
  compression @4 :Compression; # The compression used to store the key.
}

struct LweBootstrapKey {
  # A bootstrap key value is a payload and a description to interpret this payload. This structure 
  # can be used to store and communicate a bootstrap key.
  
  info @0 :LweBootstrapKeyInfo; # The description of the bootstrap key.
  payload @1 :Payload; # The payload.
}

############################################################################## LWE keyswitch keys ##

struct LweKeyswitchKeyParams {
  # A keyswitch key is parameterized by a few quantities of cryptographic importance. This structure 
  # represents those parameters.
  #
  # Note:
  #   For now, only keys with the same input and output key types can be represented. 

  levelCount @0 :UInt32; # The number of levels of the ciphertexts.
  baseLog @1 :UInt32; # The logarithm of the base of ciphertexts.
  variance @2 :Float64; # The variance used to encrypt the ciphertexts.
  integerPrecision @3 :UInt32; # The bitwidth of the integers used to store the ciphertexts.
  inputLweDimension @6 :UInt32; # The dimension of the input secret key.
  outputLweDimension @7 :UInt32; # The dimension of the output secret key.
  modulus @4 :Modulus; # The modulus used to perform operations with this key.
  keyType @5 :KeyType; # The distribution of the input and output secret keys.
}

struct LweKeyswitchKeyInfo {
  # A keyswitch key value is uniquely described by cryptographic parameters and a few application
  # related quantities. This structure represents this description of a keyswitch key.
  # 
  # Note:
  #   Keyswitch keys with same parameters, compression, input and output id, are allowed to co-exist
  #   in a program as long as they have different ids.

  id @0 :UInt32; # The identifier of the keyswitch key.
  inputId @1 :UInt32; # The identifier of the input secret key.
  outputId @2 :UInt32; # The identifier of the output secret key.
  params @3 :LweKeyswitchKeyParams; # The cryptographic parameters of the key.
  compression @4 :Compression; # The compression used to store the key.
}

struct LweKeyswitchKey {
  # A keyswitch key value is a payload and a description to interpret this payload. This structure 
  # can be used to store and communicate a keyswitch key.
  
  info @0 :LweKeyswitchKeyInfo; # The description of the keyswitch key.
  payload @1 :Payload; # The payload.
}

########################################################################## Packing keyswitch keys ##

struct PackingKeyswitchKeyParams {
  # A packing keyswitch key is parameterized by a few quantities of cryptographic importance. This 
  # structure represents those parameters.
  # 
  # Note:
  #   For now, only keys with the same input and output key types can be represented. 

  levelCount @0 :UInt32; # The number of levels of the ciphertexts.
  baseLog @1 :UInt32; # The logarithm of the base of the ciphertexts.
  glweDimension @2 :UInt32; # The glwe dimension of the ciphertexts.
  polynomialSize @3 :UInt32; # The polynomial size of the ciphertexts.
  inputLweDimension @4 :UInt32; # The input lwe dimension.
  innerLweDimension @5 :UInt32; # The intermediate lwe dimension.
  variance @6 :Float64; # The variance used to encrypt the ciphertexts.
  integerPrecision @7 :UInt32; # The bitwidth of the integers used to store the ciphertexts.
  modulus @8 :Modulus; # The modulus used to perform operations with this key.
  keyType @9 :KeyType; # The distribution of the input and output secret keys.
}

struct PackingKeyswitchKeyInfo {
  # A packing keyswitch key value is uniquely described by cryptographic parameters and a few 
  # application related quantities. This structure represents this description of a packing 
  # keyswitch key.
  # 
  # Note:
  #   Packing keyswitch keys with same parameters, compression, input and output id, are allowed to 
  #   co-exist in a program as long as they have different ids.

  id @0 :UInt32; # The identifier of the packing keyswitch key.
  inputId @1 :UInt32; # The identifier of the input secret key.
  outputId @2 :UInt32; # The identifier of the output secret key.
  params @3 :PackingKeyswitchKeyParams; # The cryptographic parameters of the key.
  compression @4 :Compression; # The compression used to store the key.
}

struct PackingKeyswitchKey {
  # A packiing keyswitch key value is a payload and a description to interpret this payload. This 
  # structure can be used to store and communicate a packing keyswitch key.
       
  info @0 :PackingKeyswitchKeyInfo; # The description of the packing keyswitch key.
  payload @1 :Payload; # The payload.
}

######################################################################################### Keysets ##

struct KeysetInfo {
  # The keyset needed for an application can be described by an ensemble of descriptions of the 
  # different keys used in the program. This structure represents such a description.

  lweSecretKeys @0 :List(LweSecretKeyInfo); # The secret key descriptions.
  lweBootstrapKeys @1 :List(LweBootstrapKeyInfo); # The bootstrap key descriptions
  lweKeyswitchKeys @2 :List(LweKeyswitchKeyInfo); # The keyswitch key descriptions.
  packingKeyswitchKeys @3 :List(PackingKeyswitchKeyInfo); # The packing keyswitch key descriptions.
}

struct ServerKeyset {
  # A server keyset is represented by an ensemble of evaluation key values. This structure allows to 
  # store and communicate such a keyset.
    
  lweBootstrapKeys @0 :List(LweBootstrapKey); # The bootstrap key values.
  lweKeyswitchKeys @1 :List(LweKeyswitchKey); # The keyswitch key values.
  packingKeyswitchKeys @2 :List(PackingKeyswitchKey); # The packing keyswitch key values.
}

struct ClientKeyset {
  # A client keyset is represented by an ensemble of secret key values. This structure allows to 
  # store and communicate such a keyset.
    
  lweSecretKeys @0 :List(LweSecretKey); # The secret key values.
}

struct Keyset {
  # A complete application keyset is the union of a server keyset, and a client keyset. This 
  # structure allows to store and communicate such a keyset.
       
  server @0 :ServerKeyset;
  client @1 :ClientKeyset;
}

####################################################################################### Encodings ##

struct EncodingInfo {
  # A value in an fhe program can encode various kind of informations, be it encrypted or not.
  # To correctly communicate, the different parties participating in the execution of the program 
  # must share informations about what encoding is used for values exchanged at their boundaries.
  # This structure represents such informations.
  # 
  # Note:
  #   The shape field is expected to contain the _abstract_ shape of the value. This means that for 
  #   an encrypted value, the shape must not contain informations about the shape of the 
  #   ciphertext(s) themselves. Said differently, the shape must be the one that would be used if 
  #   the value was not encrypted.

  shape @0 :Shape; # The shape of the value.
  encoding :union { 
    # The encoding for each scalar element of the value.

   	integerCiphertext @1 :IntegerCiphertextEncodingInfo;
   	booleanCiphertext @2 :BooleanCiphertextEncodingInfo;
   	plaintext @3 :PlaintextEncodingInfo;
   	index @4 :IndexEncodingInfo;
  }
}

struct IntegerCiphertextEncodingInfo {
  # A ciphertext can be used to represent an integer value. This structure represents the 
  # informations needed to encode such an integer.

  width @0 :UInt32; # The bitwidth of the encoded integer.
  isSigned @1 :Bool; # The signedness of the encoded integer.
  mode :union { 
    # The mode used to encode the integer.

  	native @2 :NativeMode;
	  chunked @3 :ChunkedMode;
    crt @4 :CrtMode;
  }

  struct NativeMode {
    # An integer of width from 1 to 8 bits can be encoded in a single ciphertext natively, by 
    # being shifted in the most significant bits. This structure represents this integer encoding 
    # mode.
  } 

  struct ChunkedMode {
    # An integer of width from 1 to n can be encoded in a set of ciphertexts by chunking the bits 
    # of the original integer. This structure represents this integer encoding mode.
    
	  size @0 :UInt32; # The number of chunks to be used.
	  width @1 :UInt32; # The number of bits encoded by each chunks.
  }

  struct CrtMode {
    # An integer of width 1 to 16 can be encoded in a set of ciphertexts, by decomposing a value 
    # using a set of pairwise coprimes. This structure represents this integer encoding mode.
  	
    moduli @0 :List(UInt32); # The coprimes used to decompose the original value.
  }
}

struct BooleanCiphertextEncodingInfo {
  # A ciphertext can be used to represent a boolean value. This structure represents such an 
  # encoding.
}

struct PlaintextEncodingInfo {
  # A cleartext value can be used to represent a plaintext value used in computation with 
  # ciphertexts. This structure represent such an encoding.
}

struct IndexEncodingInfo {
  # A cleartext value can be used to represent an index value used to index in a tensor of values. 
  # This structure represent such an encoding.
}

struct CircuitEncodingInfo {
  # A circuit encodings is described by the set of encodings used for its inputs and outputs and its
  # name. This structure represents this circuit encoding signature.
  #
  # Note:
  #   The order of the input and output lists matters. The order of values should be the same when 
  #   executing the circuit. Also, the name is expected to be unique in the program.
       
  inputs @0 :List(EncodingInfo); # The ordered list of input encoding infos.
  outputs @1 :List(EncodingInfo); # The ordered list of output encoding infos.
  name @2 :Text; # The name of the circuit.
}

struct ProgramEncodingInfo {
  # A program encodings is described by the set of circuit encodings. This structure represents 
  # this ensemble of encoding signatures.
       
  circuits @0 :List(CircuitEncodingInfo); # The list of the circuit encoding infos.
}

###################################################################################### Encryption ##

struct LweCiphertextEncryptionInfo {
  # The encryption of a cleartext value requires some parameters to operate. This structure 
  # represents those parameters.

  keyId @0 :UInt32; # The identifier of the secret key used to perform the encryption.
  variance @1 :Float64; # The variance of the noise injected during encryption.
  lweDimension @2 :UInt32; # The lwe dimension of the ciphertext.
  modulus @3 :Modulus; # The modulus used when performing operations on this ciphertext.
}

########################################################################################## Typing ##

struct TypeInfo{
  union {
    # The different possible type of values.

    lweCiphertext @0 :LweCiphertextTypeInfo;
    plaintext @1 :PlaintextTypeInfo;
    index @2 :IndexTypeInfo;
  }
}

struct LweCiphertextTypeInfo {
  # A ciphertext value can flow in and out of a circuit. This structure represents the informations
  # needed to verify and pre-or-post process this value.
  #
  # Note:
  #   Two shape information are carried in this type. The abstract shape is the shape the tensor 
  #   would have if the values were cleartext. That is, it does not take into account the encryption 
  #   process. The concrete shape is the final shape of the object accounting for the encryption, 
  #   that usually add one or more dimension to the object.

  abstractShape @0 :Shape; # The abstract shape of the value.
  concreteShape @1 :Shape; # The concrete shape of the value.
  integerPrecision @2 :UInt32; # The precision of the integers used for storage.
  encryption @3 :LweCiphertextEncryptionInfo; # The informations relative to the encryption.
  compression @4 :Compression; # The compression used for this value.
  encoding :union {
    # The encoding of the value stored inside the ciphertext.

  	integer @5 :IntegerCiphertextEncodingInfo;
   	boolean @6 :BooleanCiphertextEncodingInfo;
  }
}

struct PlaintextTypeInfo {
  # A plaintext value can flow in and out of a circuit. This structure represents the informations 
  # needed to verify and pre-or-post process this value.

  shape @0 :Shape; # The shape of the value.
  integerPrecision @1 :UInt32; # The precision of the integers.
  isSigned @2 :Bool; # The signedness of the integers.
}

struct IndexTypeInfo {
  # A plaintext value can flow in and out of a circuit. This structure represents the informations
  # needed to verify and pre-or-post process this value.

  shape @0 :Shape; # The shape of the value.
  integerPrecision @1 :UInt32; # The precision of the indexes.
  isSigned @2 :Bool; # The signedness of the indexes.
}
 
############################################################################### Circuit signature ##

struct GateInfo {
  # A value flowing in or out of a circuit is expected to be of a given type, according to the 
  # signature of this circuit. This structure represents such a type in a circuit signature.
  
  rawInfo @0 :RawInfo; # The raw information that raw data must be possible to parse with.
  typeInfo @1 :TypeInfo; # The type of the value expected at the gate.
}

struct CircuitInfo {
  # A circuit signature can be described completely by the type informations for its input and 
  # outputs, as well as its name. This structure regroup those informations.
  # 
  # Note:
  #   The order of the input and output lists matters. The order of values should be the same when 
  #   executing the circuit. Also, the name is expected to be unique in the program.

    inputs @0 :List(GateInfo); # The ordered list of input types.
    outputs @1 :List(GateInfo); # The ordered list of output types.
    name @2 :Text; # The name of the circuit.
}

struct ProgramInfo {
  # A complete program can be described by the ensemble of circuit signatures, and the description
  # of the keyset that go with it. This structure regroup those informations.
  
  keyset @0 :KeysetInfo; # The informations on the keyset of the program.
  circuits @1 :List(CircuitInfo); # The informations for the different circuits of the program.
}

########################################################################################## Values ##

struct Value {
  # A value is the union of a binary payload, raw informations to turn this payload into an integer
  # tensor, and typ informations to check and pre-post process values at the boundary of a 
  # circuit. This structure can be used to store, or communicate a value used during a program 
  # execution.
  # 
  # Note:
  #   The value info is a smaller runtime equivalent of the gate types used in the circuit 
  #   signatures.
  
  payload @0 :Payload; # The binary payload containing a raw integer tensor.
  rawInfo @1 :RawInfo; # The informations to parse the binary payload.
  typeInfo @2 :TypeInfo; # The type of the value.
}

################################################################################### Public values ##

struct PublicArguments {
       args @0 :List(Value);       
}

struct PublicResults {
       results @0 :List(Value);
}


