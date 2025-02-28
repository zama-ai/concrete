use serde::{Deserialize};

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
pub struct IntegerCiphertextEncodingInfo_Mode_NativeMode();

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
pub struct BooleanCiphertextEncodingInfo();
