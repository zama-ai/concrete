const SECURITY_WEIGHTS_ARRAY: [(u64, SecurityWeights); 4] = [
    (80, SecurityWeights { slope: -0.0404263311936459, bias: 1.660978864143658, minimal_lwe_dimension: 450 }),
    (112, SecurityWeights { slope: -0.02967013708113588, bias: 2.16246371408387, minimal_lwe_dimension: 450 }),
    (128, SecurityWeights { slope: -0.026405028765226296, bias: 2.482642269104389, minimal_lwe_dimension: 450 }),
    (192, SecurityWeights { slope: -0.018610403247590064, bias: 3.2996236848399008, minimal_lwe_dimension: 606 }),
];
