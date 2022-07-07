# What is Concrete?

<mark style="background-color:yellow;">⭐️</mark> [<mark style="background-color:yellow;">Star the repo on Github</mark>](https://github.com/zama-ai/concrete) <mark style="background-color:yellow;">| 🗣</mark> [<mark style="background-color:yellow;">Community support forum</mark> ](https://community.zama.ai)<mark style="background-color:yellow;">| 📁</mark> [<mark style="background-color:yellow;">Contribute to the project</mark>](https://docs.zama.ai/concrete/developers/contributing)<mark style="background-color:yellow;"></mark>

![](_static/zama\_concrete\_docs\_home.jpg)

`concrete` is a Rust crate (library) meant to abstract away the details of Fully Homomorphic Encryption (FHE) to enable non-cryptographers to build applications that use FHE.

FHE is a powerful cryptographic tool, which allows computation to be performed directly on encrypted data without needing to decrypt it first.

{% hint style="warning" %}
Concrete 0.2 is a complete rewrite of the Concrete library. Previous release (0.1.x) was cryptography oriented while the new 0.2 version is developer oriented. There is no backward compatibility.
{% endhint %}

### Key Cryptographic concepts

Concrete Library implements Zama’s variant of Fully Homomorphic Encryption over the Torus (TFHE). TFHE is based on Learning With Errors (LWE), a well studied cryptographic primitive believed to be secure even against quantum computers.

In cryptography, a raw value is called a message (also sometimes called a cleartext), an encoded message is called a plaintext and an encrypted plaintext is called a ciphertext.

The idea of homomorphic encryption is that you can compute on ciphertexts while not knowing messages encrypted in them. A scheme is said to be _fully homomorphic_, meaning any program can be evaluated with it, if at least two of the following operations are supported \($$x$$is a plaintext and $$E[x]$$ is the corresponding ciphertext\):

* homomorphic univariate function evaluation: $$f(E[x]) = E[f(x)]$$
* homomorphic addition: $$E[x] + E[y] = E[x + y]$$
* homomorphic multiplication: $$E[x] * E[y] = E[x * y]$$

Zama's variant of TFHE is fully homomorphic and deals with fixed-precision numbers as messages. It implements homomorphic addition and function evaluation via **Programmable Bootstrapping**. You can read more about Zama's TFHE variant in the [preliminary whitepaper](https://whitepaper.zama.ai/).

Using FHE in a Rust program with Concrete consists in:

* generating a secret client and a server key using secure parameters
* encrypting plaintexts using the secret key to produce ciphertexts
* operating homomorphically on ciphertexts with the server key
* decrypting the resulting ciphertexts into plaintexts using the secret key

If you would like to know more about the problems that FHE solves, we suggest you review our [6 minute introduction to homomorphic encryption](https://6min.zama.ai/).

## Relationship between Concrete crates

This crate provides different types which are the counterparts of native Rust types (such as `bool`, `u8`, `u16`) in the FHE domain.

With `concrete` crate, our goal is to let any developer without any prior cryptographic knowledge to build his own FHE application. To reach that goal, some of the complexity is hidden from the user.

Aside from the advanced customization options offered directly by `concrete`, an advanced user could also have a look at the underlying libraries.

{% hint style="info" %}
`concrete` is built as a framework of libraries, but we greatly suggest to any user to start building applications with the `concrete` crate.
{% endhint %}

In its current state, `concrete` crate is built on top of 3 primitive crate types: respectively, 
`concrete-boolean` for boolean type, `concrete-shortint` for the integers from 2 to 7 bits, and `concrete-int` for the integer from 4 to 16 bits. Cryptographic operations will be handled by `concrete-core`.&#x20;

We have summarized the relation between all `concrete` crates with the following diagram:

![](\_static/concrete\_libs.png)
