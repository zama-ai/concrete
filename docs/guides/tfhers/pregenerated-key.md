# Scenario 2: Pregenerated TFHE-rs keys

This document provides an overview of the second use case for combining TFHE-rs and Concrete, where two parties work with pregenerated TFHE-rs keysets. The focus is on how to perform computations on encrypted TFHE-rs values using Concrete, without the need for a shared secret key.

This scenario is interesting in the case of threshold cryptography, when the secret key is not known by a single party but split into different parties, and when the secret key cannot be changed (e.g. because it's a permanent key of an encrypted blockchain).

This use case is not managed currently by Concrete and will be available later.

In this scenario, two parties have a predetermined set of TFHE-rs keysets. The goal is to compute on encrypted TFHE-rs values in Concrete, but without sharing any secret key. Instead, the party using Concrete will rely only on TFHE-rs public keys and will optimize its operations according to the parameters associated with those keys. Meanwhile, the party using TFHE-rs will have the capacity to perform encryption, decryption, and computation.

## Workflow

The workflow for this scenario is as follows:

1. We generate multiple keysets (including secret keys and public keys) with different sets of parameters.
2. Considering the public keys, Concrete's Optimizer will limit its parameter search space to find the best configuration that is compatible with the existing TFHE-rs keys.

![alt text](../../_static/tfhers/tfhers_uc2.excalidraw.svg)

## Use case

This approach is particularly useful in scenarios involving threshold cryptography, where the secret key is distributed across multiple parties. This setup is beneficial when the secret key cannot be modified, such as in situations where the key is a permanent fixture, like in encrypted blockchains.

## Future Support

Currently, this use case is not supported by Concrete but will be available later.
