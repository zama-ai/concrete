// RUN: concretecompiler --action=roundtrip %s  2>&1| FileCheck %s
module attributes {
    // CHECK: glwe.sk_001_binary_distribution = #glwe.secret_key_distribution<Binary>
    // CHECK: glwe.sk_002_ternary_distribution = #glwe.secret_key_distribution<Ternary>
    // CHECK: glwe.sk_003_custom_distribution = #glwe.secret_key_distribution<average_mean=#glwe.expr<1.000000e-01>, average_variance=#glwe.expr<2.000000e-01>, kind=MyCustomDistribution>
    
    // Define a secret key distribution with a binary distribution
    glwe.sk_001_binary_distribution = #glwe.secret_key_distribution<Binary>,
    // Define a secret key distribution with a ternary distribution
    glwe.sk_002_ternary_distribution = #glwe.secret_key_distribution<Ternary>,
    // Define a secret key distribution with a custom distribution
    glwe.sk_003_custom_distribution = #glwe.secret_key_distribution<
        average_mean = <0.1>,
        average_variance = <0.2>,
        kind = MyCustomDistribution
    >
} {

}
