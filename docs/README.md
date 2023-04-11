# What is Concrete?

📁 [Github](https://github.com/zama-ai/concrete) | 💛 [Community support](https://zama.ai/community) | 🟨 [Zama Bounty Program](https://github.com/zama-ai/bounty-program)

<figure><img src="_static/zama_home_docs.png" alt=""><figcaption></figcaption></figure>

**Concrete** is an open-source framework which simplifies the use of fully homomorphic encryption (FHE).

FHE is a powerful cryptographic tool, which allows computation to be performed directly on encrypted data without needing to decrypt it first. With FHE, you can build services that preserve privacy for all users. FHE is also great against data breaches as everything is done on encrypted data. Even if the server is compromised, in the end no sensitive data is leaked.

Since writing FHE program is hard, concrete framework contains a TFHE Compiler based on LLVM to make this process easier for developers.

## Organization of this documentation

This documentation is split into several sections:

* **Getting Started** gives you the basics,
* **Tutorials** gives you some essential examples on various features of the library,
* **How to** helps you perform specific tasks,
* **Developer** explains the inner workings of the library and everything related to contributing to the project.

## Looking for support? Ask our team!

* Support forum: [https://community.zama.ai](https://community.zama.ai) (we answer in less than 24 hours).
* Live discussion on the FHE.org discord server: [https://discord.fhe.org](https://discord.fhe.org) (inside the #**concrete** channel).
* Do you have a question about Zama? You can write us on [Twitter](https://twitter.com/zama\_fhe) or send us an email at: **hello@zama.ai**

## How is it different from Concrete Numpy?

Concrete Numpy was the former name of Concrete Compiler Python's frontend. Starting from v1, Concrete Compiler is now open sourced and the package name is updated from `concrete-numpy` to `concrete-python` (as `concrete` is already booked for a non FHE-related project).

Users from Concrete-Numpy could safely update to Concrete with few changes explained in the [upgrading document](https://github.com/zama-ai/concrete/blob/main/UPGRADING.md).

## How is it different from the previous Concrete (v0.x)?

Before v1.0, Concrete was a set of Rust libraries implementing Zama's variant of TFHE. Starting with v1, Concrete is now Zama's TFHE compiler framework only. Rust library could be found under [TFHE-rs](https://github.com/zama-ai/tfhe-rs) project.
