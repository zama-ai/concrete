#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_ENUMS
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_ENUMS

include "mlir/IR/EnumAttr.td"

def GLWE_SecretKeyDistributionKind: I32EnumAttr<"SecretKeyDistributionKind", "Secret key distribution kind", [
    I32EnumAttrCase<"Binary", 0>,
    I32EnumAttrCase<"Ternary", 1>,
    I32EnumAttrCase<"Custom", 2>,
]> {
    let cppNamespace = "::mlir::concretelang::GLWE";
    let genSpecializedAttr = 0;
}


def GLWE_NoiseDistribution: I32EnumAttr<"NoiseDistribution", "Noise distribution", [
    I32EnumAttrCase<"Gaussian", 0>,
    I32EnumAttrCase<"TUniform", 1>
]> {
    let cppNamespace = "::mlir::concretelang::GLWE";
    let genSpecializedAttr = 0;
}

#endif
