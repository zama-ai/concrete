#![allow(non_camel_case_types, non_snake_case, unused)]
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ProgramInfo{
    pub circuits: Vec<CircuitInfo>
}

#[derive(Debug, Deserialize)]
pub struct CircuitInfo{
    pub inputs: Vec<GateInfo>,
    pub outputs: Vec<GateInfo>,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct GateInfo{
    pub rawInfo: RawInfo,
    pub typeInfo: TypeInfo
}

#[derive(Debug, Deserialize)]
pub struct RawInfo{
    pub shape: Shape,
    pub integerPrecision: u32,
    pub isSigned: bool,
}

#[derive(Debug, Deserialize)]
pub struct Shape{
    pub dimensions: Vec<u32>
}


#[derive(Debug, Deserialize)]
pub enum TypeInfo{
    lweCiphertext(LweCiphertextTypeInfo),
    plaintext(PlaintextTypeInfo),
    index(IndexTypeInfo)
}

#[derive(Debug, Deserialize)]
pub struct PlaintextTypeInfo{
    pub shape: Shape,
    pub integerPrecision: u32,
    pub isSigned: bool
}

#[derive(Debug, Deserialize)]
pub struct IndexTypeInfo{
    pub shape: Shape,
    pub integerPrecision: u32,
    pub isSigned: bool
}

#[derive(Debug, Deserialize)]
pub struct LweCiphertextTypeInfo{
    pub abstractShape: Shape,
    pub concreteShape: Shape,
    pub integerPrecision: u32,
    pub encryption: LweCiphertextEncryptionInfo,
    pub compression: Compression,
    pub encoding: LweCiphretextTypeInfo_Encoding
}

#[derive(Debug, Deserialize)]
pub struct LweCiphertextEncryptionInfo{
    pub keyId: u32,
    pub variance: f64,
    pub lweDimension: u32,
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
    pub width: u32,
    pub isSigned: bool,
    pub mode: IntegerCiphertextEncodingInfo_Mode
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
    pub size: u32,
    pub width: u32
}

#[derive(Debug, Deserialize)]
pub struct IntegerCiphertextEncodingInfo_Mode_CrtMode{
    pub moduli: Vec<u32>
}

#[derive(Debug, Deserialize)]
pub struct BooleanCiphertextEncodingInfo{}

#[derive(Debug, Deserialize)]
pub enum KeyType{
    binary = 0,
    ternary = 1,
}

#[derive(Debug, Deserialize)]
pub struct Modulus{
    pub modulus: Modulus_enum
}

#[derive(Debug, Deserialize)]
pub enum Modulus_enum{
    native(NativeModulus),
    powerOfTwo(PowerOfTwoModulus),
    integer(IntegerModulus),
}

#[derive(Debug, Deserialize)]
pub struct NativeModulus{}


#[derive(Debug, Deserialize)]
pub struct PowerOfTwoModulus{
    pub power: u32,
}

#[derive(Debug, Deserialize)]
pub struct IntegerModulus{
    pub modulus: u32,
}

#[derive(Debug, Deserialize)]
pub struct LweSecretKeyInfo{
    pub id: u32,
    pub params: LweSecretKeyParams
}

#[derive(Debug, Deserialize)]
pub struct LweSecretKeyParams{
    pub lweDimension: u32,
    pub integerPrecision: u32,
    pub keyType: KeyType
}

#[derive(Debug, Deserialize)]
pub struct LweKeyswitchKeyInfo{
    pub id: u32,
    pub inputId: u32,
    pub outputId: u32,
    pub params: LweKeyswitchKeyParams,
    pub compression: Compression
}

#[derive(Debug, Deserialize)]
pub struct LweKeyswitchKeyParams{
    pub levelCount: u32,
    pub baseLog: u32,
    pub variance: f64,
    pub integerPrecision: u32,
    pub inputLweDimension: u32,
    pub outputLweDimension: u32,
    pub modulus: Modulus,
    pub keyType: KeyType
}

#[derive(Debug, Deserialize)]
pub struct PackingKeyswitchKeyInfo{
    pub id: u32,
    pub inputId: u32,
    pub outputId: u32,
    pub params: PackingKeyswitchKeyParams,
    pub compression: Compression
}

#[derive(Debug, Deserialize)]
pub struct PackingKeyswitchKeyParams{
    pub levelCount: u32,
    pub baseLog: u32,
    pub glweDimension: u32,
    pub polynomialSize: u32,
    pub inputLweDimension: u32,
    pub innerLweDimension: u32,
    pub variance: f64,
    pub integerPrecision: u32,
    pub modulus: Modulus,
    pub keyType: KeyType
}

#[derive(Debug, Deserialize)]
pub struct LweBootstrapKeyInfo{
    pub id: u32,
    pub inputId: u32,
    pub outputId: u32,
    pub params: LweBootstrapKeyParams,
    pub compression: Compression
}

#[derive(Debug, Deserialize)]
pub struct LweBootstrapKeyParams{
    pub levelCount: u32,
    pub baseLog: u32,
    pub glweDimension: u32,
    pub polynomialSize: u32,
    pub inputLweDimension: u32,
    pub variance: f64,
    pub integerPrecision: u32,
    pub modulus: Modulus,
    pub keyType: KeyType
}

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
