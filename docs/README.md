# What is Concrete?

📁 [Github](https://github.com/zama-ai/concrete) | 💛 [Community support](https://zama.ai/community) | 🟨 [Zama Bounty Program](https://github.com/zama-ai/bounty-program)

<figure><img src="_static/zama_home_docs.png" alt=""><figcaption></figcaption></figure>

**Concrete** is an open source framework which simplifies the use of Fully Homomorphic Encryption (FHE).

FHE is a powerful cryptographic tool, allowing computation to be performed directly on encrypted data without needing to decrypt it. With FHE, you can build services that preserve privacy for all users. FHE also offers ideal protection against data breaches as everything is done on encrypted data. Even if the server is compromised, no sensitive data is leaked.

Since writing FHE programs is a difficult task, Concrete framework contains a TFHE Compiler based on [LLVM](https://en.wikipedia.org/wiki/LLVM) to make this process easier for developers.

## Organization of this documentation

This documentation is split into several sections:

* **Getting Started** gives you the basics,
* **Tutorials** provides essential examples on various features of the library,
* **How to** helps you perform specific tasks,
* **Developer** explains the inner workings of the library and everything related to contributing to the project.

## Looking for support? Ask our team!

* Support forum: [https://community.zama.ai](https://community.zama.ai) (we answer in less than 24 hours).
* Live discussion on the FHE.org discord server: [https://discord.fhe.org](https://discord.fhe.org) (inside the #**concrete** channel).
* Do you have a question about Zama? Write us on [Twitter](https://twitter.com/zama\_fhe) or send us an email at: **hello@zama.ai**

## How is Concrete different from Concrete Numpy?

Concrete Numpy was the former name of the Python frontend of the Concrete Compiler. Concrete Compiler is now open source, and the package name is updated from `concrete-numpy` to `concrete-python` (as `concrete` is already booked for a non FHE-related project).

Users from Concrete Numpy can safely update to Concrete, with a few required changes, as explained in the [upgrading document](https://github.com/zama-ai/concrete/blob/main/UPGRADING.md).

## How is it different from the previous version of Concrete?

Before v1.0, Concrete was a set of Rust libraries implementing Zama's variant of TFHE. Starting with v1, Concrete is now Zama's TFHE Compiler framework only. The Rust library is now called [TFHE-rs](https://github.com/zama-ai/tfhe-rs).
