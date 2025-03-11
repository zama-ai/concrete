#![allow(non_camel_case_types, non_snake_case)]
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ProgramInfo{
    circuits: Vec<CircuitInfo>
}

#[derive(Debug, Deserialize)]
pub struct CircuitInfo{
    inputs: Vec<GateInfo>,
    outputs: Vec<GateInfo>,
    name: String,
}

#[derive(Debug, Deserialize)]
pub struct GateInfo{
    rawInfo: RawInfo,
    typeInfo: TypeInfo
}

#[derive(Debug, Deserialize)]
pub struct RawInfo{
    shape: Shape,
    integerPrecision: u32,
    isSigned: bool,
}

#[derive(Debug, Deserialize)]
pub struct Shape{
    dimensions: Vec<u32>
}


#[derive(Debug, Deserialize)]
pub enum TypeInfo{
    lweCiphertext(LweCiphertextTypeInfo),
    plaintext(PlaintextTypeInfo),
    index(IndexTypeInfo)
}

#[derive(Debug, Deserialize)]
pub struct PlaintextTypeInfo{
    shape: Shape,
    integerPrecision: u32,
    isSigned: bool
}

#[derive(Debug, Deserialize)]
pub struct IndexTypeInfo{
    shape: Shape,
    integerPrecision: u32,
    isSigned: bool
}

#[derive(Debug, Deserialize)]
pub struct LweCiphertextTypeInfo{
    abstractShape: Shape,
    concreteShape: Shape,
    integerPrecision: u32,
    encryption: LweCiphertextEncryptionInfo,
    compression: Compression,
    encoding: LweCiphretextTypeInfo_Encoding
}

#[derive(Debug, Deserialize)]
pub struct LweCiphertextEncryptionInfo{
    keyId: u32,
    variance: f64,
    lweDimension: u32,
}

#[derive(Debug, Deserialize)]
pub enum Compression {
    none,
    seed,
    paillier
}

#[derive(Debug, Deserialize)]
pub enum LweCiphretextTypeInfo_Encoding{
    integer(IntegerCiphertextEncodingInfo),
    boolean(BooleanCiphertextEncodingInfo)
}

#[derive(Debug, Deserialize)]
pub struct IntegerCiphertextEncodingInfo{
    width: u32,
    isSigned: bool,
    mode: IntegerCiphertextEncodingInfo_Mode
}

#[derive(Debug, Deserialize)]
pub enum IntegerCiphertextEncodingInfo_Mode{
    native(IntegerCiphertextEncodingInfo_Mode_NativeMode),
    chunked(IntegerCiphertextEncodingInfo_Mode_ChunkedMode),
    crt(IntegerCiphertextEncodingInfo_Mode_CrtMode),
}

#[derive(Debug, Deserialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_NativeMode{}

#[derive(Debug, Deserialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_ChunkedMode{
    size: u32,
    width: u32
}

#[derive(Debug, Deserialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_CrtMode{
    moduli: Vec<u32>
}

#[derive(Debug, Deserialize)]
pub struct BooleanCiphertextEncodingInfo{}

#[cfg(test)]
mod test{
    use super::ProgramInfo;

    const TEST_JSON: &'static str = "
    {
      \"keyset\": {
        \"lweSecretKeys\": [
          { \"id\": 0, \"params\": { \"lweDimension\": 6144, \"integerPrecision\": 64, \"keyType\": \"binary\" } },
          { \"id\": 1, \"params\": { \"lweDimension\": 865, \"integerPrecision\": 64, \"keyType\": \"binary\" } }
        ],
        \"lweBootstrapKeys\": [
          {
            \"id\": 0,
            \"inputId\": 1,
            \"outputId\": 0,
            \"params\": {
              \"levelCount\": 1,
              \"baseLog\": 23,
              \"glweDimension\": 3,
              \"polynomialSize\": 2048,
              \"variance\": 4.70197740328915e-38,
              \"integerPrecision\": 64,
              \"modulus\": { \"modulus\": { \"native\": {} } },
              \"keyType\": \"binary\",
              \"inputLweDimension\": 865
            },
            \"compression\": \"none\"
          }
        ],
        \"lweKeyswitchKeys\": [
          {
            \"id\": 0,
            \"inputId\": 0,
            \"outputId\": 1,
            \"params\": {
              \"levelCount\": 4,
              \"baseLog\": 4,
              \"variance\": 1.692989133024556e-12,
              \"integerPrecision\": 64,
              \"modulus\": { \"modulus\": { \"native\": {} } },
              \"keyType\": \"binary\",
              \"inputLweDimension\": 6144,
              \"outputLweDimension\": 865
            },
            \"compression\": \"none\"
          }
        ],
        \"packingKeyswitchKeys\": []
      },
      \"circuits\": [
        {
          \"inputs\": [
            {
              \"rawInfo\": {
                \"shape\": { \"dimensions\": [6145] },
                \"integerPrecision\": 64,
                \"isSigned\": false
              },
              \"typeInfo\": {
                \"lweCiphertext\": {
                  \"abstractShape\": { \"dimensions\": [] },
                  \"concreteShape\": { \"dimensions\": [6145] },
                  \"integerPrecision\": 64,
                  \"encryption\": {
                    \"keyId\": 0,
                    \"variance\": 4.70197740328915e-38,
                    \"lweDimension\": 6144,
                    \"modulus\": { \"modulus\": { \"native\": {} } }
                  },
                  \"compression\": \"none\",
                  \"encoding\": { \"integer\": { \"width\": 5, \"isSigned\": false, \"mode\": { \"native\": {} } } }
                }
              }
            }
          ],
          \"outputs\": [
            {
              \"rawInfo\": {
                \"shape\": { \"dimensions\": [6145] },
                \"integerPrecision\": 64,
                \"isSigned\": false
              },
              \"typeInfo\": {
                \"lweCiphertext\": {
                  \"abstractShape\": { \"dimensions\": [] },
                  \"concreteShape\": { \"dimensions\": [6145] },
                  \"integerPrecision\": 64,
                  \"encryption\": {
                    \"keyId\": 0,
                    \"variance\": 4.70197740328915e-38,
                    \"lweDimension\": 6144,
                    \"modulus\": { \"modulus\": { \"native\": {} } }
                  },
                  \"compression\": \"none\",
                  \"encoding\": { \"integer\": { \"width\": 5, \"isSigned\": false, \"mode\": { \"native\": {} } } }
                }
              }
            }
          ],
          \"name\": \"dec\"
        },
        {
          \"inputs\": [
            {
              \"rawInfo\": {
                \"shape\": { \"dimensions\": [6145] },
                \"integerPrecision\": 64,
                \"isSigned\": false
              },
              \"typeInfo\": {
                \"lweCiphertext\": {
                  \"abstractShape\": { \"dimensions\": [] },
                  \"concreteShape\": { \"dimensions\": [6145] },
                  \"integerPrecision\": 64,
                  \"encryption\": {
                    \"keyId\": 0,
                    \"variance\": 4.70197740328915e-38,
                    \"lweDimension\": 6144,
                    \"modulus\": { \"modulus\": { \"native\": {} } }
                  },
                  \"compression\": \"none\",
                  \"encoding\": { \"integer\": { \"width\": 5, \"isSigned\": false, \"mode\": { \"native\": {} } } }
                }
              }
            }
          ],
          \"outputs\": [
            {
              \"rawInfo\": {
                \"shape\": { \"dimensions\": [6145] },
                \"integerPrecision\": 64,
                \"isSigned\": false
              },
              \"typeInfo\": {
                \"lweCiphertext\": {
                  \"abstractShape\": { \"dimensions\": [] },
                  \"concreteShape\": { \"dimensions\": [6145] },
                  \"integerPrecision\": 64,
                  \"encryption\": {
                    \"keyId\": 0,
                    \"variance\": 4.70197740328915e-38,
                    \"lweDimension\": 6144,
                    \"modulus\": { \"modulus\": { \"native\": {} } }
                  },
                  \"compression\": \"none\",
                  \"encoding\": { \"integer\": { \"width\": 5, \"isSigned\": false, \"mode\": { \"native\": {} } } }
                }
              }
            }
          ],
          \"name\": \"inc\"
        }
      ]
    }
    ";

   #[test]
   fn test() {
       let pi: ProgramInfo = serde_json::from_str(TEST_JSON).unwrap();
       dbg!(pi);
   }
}
