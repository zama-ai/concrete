<p align="center">
<!-- product name logo -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/zama-ai/concrete/assets/157474013/d4680525-2371-454e-97d3-ba39c809a074">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/zama-ai/concrete/assets/157474013/95d02fb1-de48-4bb7-b175-d536bb13091c">
  <img width=600 alt="Zama Concrete">
</picture>
</p>
<hr/>

<p align="center">
  <a href="https://docs.zama.ai/concrete"> 📒 Documentation</a> | <a href="https://zama.ai/community"> 💛 Community support</a> | <a href="https://github.com/zama-ai/awesome-zama"> 📚 FHE resources by Zama</a>
</p>

<p align="center">
  <a href="https://github.com/zama-ai/concrete/releases"><img src="https://img.shields.io/github/v/release/zama-ai/concrete?style=flat-square"></a>
  <a href="https://github.com/zama-ai/concrete/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-BSD--3--Clause--Clear-%23ffb243?style=flat-square"></a>
  <a href="https://github.com/zama-ai/bounty-program"><img src="https://img.shields.io/badge/Contribute-Zama%20Bounty%20Program-%23ffd208?style=flat-square"></a>
  <a href="https://slsa.dev"><img alt="SLSA 3" src="https://slsa.dev/images/gh-badge-level3.svg" /></a>
</p>



## About

### What is Concrete
**Concrete** is an open-source FHE Compiler that simplifies the use of fully homomorphic encryption (FHE). Built on TFHE technology and leveraging LLVM, Concrete makes writing FHE programs accessible to developers without deep cryptography expertise.

#### Key Features
- 🚀 Simple Python API for FHE operations
- 🔒 Built-in security guarantees
- ⚡ GPU acceleration support
- 🛠 Automatic parameter selection
- 📊 Built-in performance analysis tools
- 🔄 Seamless integration with existing Python code

Fully Homomorphic Encryption (FHE) enables performing computations on encrypted data directly without the need to decrypt it first. FHE allows developers to build services that ensure privacy for all users. FHE is also an excellent solution against data breaches as everything is performed on encrypted data. Even if the server is compromised, no sensitive data will be leaked.

Concrete is a versatile library that can be used for a variety of purposes. For instance, [Concrete ML](https://github.com/zama-ai/concrete-ml) is built on top of Concrete to simplify Machine-Learning oriented use cases.
<br></br>



## Table of Contents
- **[Getting Started](#getting-started)**
   - [Installation](#installation)
   - [A simple example](#a-simple-example)
- **[Resources](#resources)**
   - [Concrete deep dive](#concrete-deep-dive)
   - [Tutorials](#tutorials)
   - [Documentation](#documentation)
- **[Working with Concrete](#working-with-concrete)**
   - [Citations](#citations)
   - [Contributing](#contributing)
   - [License](#license)
- **[Support](#support)**
<br></br>


## Getting Started

### System Requirements
- Python 3.9 - 3.12
- 8GB RAM minimum (16GB recommended)
- x86_64 or ARM64 processor

###  Installation
We provide multiple installation methods to suit your needs:

#### 1. PyPI Installation (Recommended)
For CPU-only version:
```shell
pip install -U pip wheel setuptools
pip install concrete-python
```

For GPU-accelerated version:
```shell
pip install -U pip wheel setuptools
pip install concrete-python --index-url https://pypi.zama.ai/gpu
```

#### 2. Docker Installation
```shell
# CPU version
docker pull zamafhe/concrete-python:v2.0.0

# GPU version
docker pull zamafhe/concrete-python:v2.0.0-gpu
```

#### Version Matrix

| Concrete Version | Python Version | CUDA Support |
|-----------------|----------------|--------------|
| 2.11.0          | 3.9 - 3.12     | ≥ 11.8      |
| 2.0.0           | 3.8 - 3.12     | ≥ 11.8      |
| 1.1.0           | 3.8 - 3.10     | ≥ 11.7      |

*Find more detailed installation instructions in [this part of the documentation](https://docs.zama.ai/concrete/getting-started/installing)*

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

### A simple example
To compute on encrypted data, you first need to define the function you want to compute, then compile it into a Concrete Circuit, which you can use to perform homomorphic evaluation.
Here is the full example:

```python
from concrete import fhe

def add(x, y):
    return x + y

compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]

print(f"Compilation...")
circuit = compiler.compile(inputset)

print(f"Key generation...")
circuit.keygen()

print(f"Homomorphic evaluation...")
encrypted_x, encrypted_y = circuit.encrypt(2, 6)
encrypted_result = circuit.run(encrypted_x, encrypted_y)
result = circuit.decrypt(encrypted_result)

assert result == add(2, 6)
```
*This example is explained in more detail [in this part of the documentation](https://docs.zama.ai/concrete/get-started/quick_start).*

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

## Resources

### Concrete deep dive
- [Part I: Concrete, Zama's Fully Homomorphic Compiler](https://www.zama.ai/post/zama-concrete-fully-homomorphic-encryption-compiler)
- [Part II: The Architecture of Concrete, Zama's Fully Homomorphic Encryption Compiler Leveraging MLIR](https://www.zama.ai/post/the-architecture-of-concrete-zama-fully-homomorphic-encryption-compiler-leveraging-mlir)
<br></br>

### Tutorials
- [[Video tutorial] Dive into Concrete - Zama's Fully Homomorphic Encryption Compiler](https://www.zama.ai/post/video-tutorial-dive-into-concrete-zamas-fully-homomorphic-encryption-compiler)
- [[Video tutorial] How To Get Started With Concrete - Zama's Fully Homomorphic Encryption Compiler](https://www.zama.ai/post/how-to-started-with-concrete-zama-fully-homomorphic-encryption-compiler)
- [The Encrypted Game of Life in Python Using Concrete](https://www.zama.ai/post/the-encrypted-game-of-life-using-concrete-python)
- [Encrypted Key-value Database Using Homomorphic Encryption](https://www.zama.ai/post/encrypted-key-value-database-using-homomorphic-encryption)

*Explore more useful resources in [Concrete tutorials](https://docs.zama.ai/concrete/v/main-1/tutorials/see-all-tutorials) and [Awesome Zama repo](https://github.com/zama-ai/awesome-zama?tab=readme-ov-file#concrete). If you have built awesome projects using Concrete, please let us know and we will be happy to showcase them here!*



### Documentation

Full, comprehensive documentation is available at [https://docs.zama.ai/concrete](https://docs.zama.ai/concrete).

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>



## Working with Concrete

### Citations
To cite Concrete in academic papers, please use the following entry:

```text
@Misc{Concrete,
  title={{Concrete: TFHE Compiler that converts python programs into FHE equivalent}},
  author={Zama},
  year={2022},
  note={\url{https://github.com/zama-ai/concrete}},
}
```
### Contributing

There are two ways to contribute to Concrete. You can:
- [Open issues](https://github.com/zama-ai/concrete/issues/new/choose) to report bugs and typos, or to suggest new ideas
- Request to become an official contributor by emailing hello@zama.ai.

Becoming an approved contributor involves signing our Contributor License Agreement (CLA). Only approved contributors can send pull requests (PRs), so get in touch before you do!

Additionally, you can contribute to advancing the FHE space with Zama by participating in our [Bounty Program and Grant Programs](https://github.com/zama-ai/bounty-and-grant-program)!
<br></br>

### License
This software is distributed under the **BSD-3-Clause-Clear** license. Read [this](LICENSE.txt) for more details.

#### FAQ

**Is Zama’s technology free to use?**
>Zama’s libraries are free to use under the BSD 3-Clause Clear license only for development, research, prototyping, and experimentation purposes. However, for any commercial use of Zama's open source code, companies must purchase Zama’s commercial patent license.
>
>Everything we do is open source and we are very transparent on what it means for our users, you can read more about how we monetize our open source products at Zama in [this blog post](https://www.zama.ai/post/open-source).

**What do I need to do if I want to use Zama’s technology for commercial purposes?**
>To commercially use Zama’s technology you need to be granted Zama’s patent license. Please contact us at hello@zama.ai for more information.

**Do you file IP on your technology?**
>Yes, all Zama’s technologies are patented.

**Can you customize a solution for my specific use case?**
>We are open to collaborating and advancing the FHE space with our partners. If you have specific needs, please email us at hello@zama.ai.

<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

## Support

<a target="_blank" href="https://community.zama.ai/c/concrete/7">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/zama-ai/concrete/assets/157474013/204c349f-b9c7-41d6-b93a-48ecd6977ff6">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/zama-ai/concrete/assets/157474013/588b6aae-9677-463a-8542-209bb8106366">
  <img alt="Support">
</picture>
</a>


🌟 If you find this project helpful or interesting, please consider giving it a star on GitHub! Your support helps to grow the community and motivates further development.


<p align="right">
  <a href="#about" > ↑ Back to top </a>
</p>

