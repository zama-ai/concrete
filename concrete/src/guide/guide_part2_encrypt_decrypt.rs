//! Part 2 of the Concrete guide where you learn how to encode, encrypt, decrypt LWE and RLWE ciphertexts
//!
//! This guide assumes that your already know how to **import the Concrete library** in your project.
//! If not, [this part](super::guide_part1_install) is designed for you ;-)
//!
//! # Step 1: Crypto overview
//!
//! With the Concrete library, you will be able to use an **FHE cryptosystem called Concrete** which is variation of [TFHE](https://eprint.iacr.org/2018/421.pdf).
//! FHE stands for Fully [Homomorphic Encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption) and roughly refers to encryption schemes where it is **possible to publicly add (resp. multiply)** a ciphertext of a message m with a ciphertext of a message m' so the result is a ciphertext of the sum m+m' (resp. the product m*m').
//!
//! In this guide we will explain how to easily **encode** a message into a **plaintext** before **encrypting** it.
//! It implies two things:
//! - define a way to encode
//! - generate a secret key
//!
//! Finally we will show how to **decrypt** a ciphertext.
//!
//! With Concrete it is possible to encrypt plaintexts inside what is called **ciphertexts**.
//! There are **two types** of ciphertexts: **LWE** and **RLWE**.
//! We will show you how to use them and will give more details about them.
//!
//! <img vspace="15" align="right" width="50" height="10" src="https://zama.ai/wp-content/uploads/2020/07/logo-black.png">
//!
//! # Step 2: Let's encode some messages
//!
//! As we mentioned above, we have to **encode** a message into **plaintexts** before being able to encrypt it.
//! To do so, we need to **select an interval** [min,max] of the **real numbers** to work with.
//! Obviously, the interval is represented with **a finite number of bits** that we also have to specify.
//! This number of bit of precision will always be preserved from the noise so your computation will be correct up to that precision.
//!
//! Note that if during an homomorphic addition or multiplication we go **above max** (resp. below min) we will go back to min (resp. max) and end up with **a value that does not make sens**.
//! However, if you're computing modular arithmetic with a power of 2 as a modulus, you wont be bothered by that, and you can use for instance the interval [0,256] with 8 bits of precision.
//!
//! Let say we want to work in [-10,10], and that 8 bits of precision are enough for our use case, the following piece of code explain how to **generate an [Encoder](super::super::crypto_api::Encoder)** for this interval.
//! Note that is this case the smallest real we can represent is 0.078125, which is the **granularity of our interval** with 8 bits of precision (max-min)/2**nb_bit_precision and this value can be computed with [get_granularity](super::super::crypto_api::Encoder::get_granularity) function.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // create an Encoder instance where messages are in the interval [-10, 10] with 8 bits of precision
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // don't hesitate to print and checkout for yourself
//!     println!("{}",encoder);
//! }
//! ```
//!
//! As you can see, we use the **[Encoder](super::super::crypto_api::Encoder) structure** and its **[new](super::super::crypto_api::Encoder::new) function** that takes as **parameter min and max** describing the real interval [min,max], and the number of bits of precision, meaning that our interval will be represented with this number of bits, and finally a number of bits for padding (which would be useful later).
//!
//! Note that working in [min,max] with a padding of one is **equivalent** as working in [min,max*2] and knowing that your messages are all inferior to max.
//! It is simply a way look at plaintexts that can help us out with homomorphic computation.
//!
//! There is also **another way** to describe that interval: the center value of the interval and a radius.
//! Here is a piece of code that instantiate the exact same [Encoder](super::super::crypto_api::Encoder) but with the [new_centered](super::super::crypto_api::Encoder::new_centered) function.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // create an Encoder instance where messages are in the interval [-10, 10] with 8 bits of precision
//!     let encoder = Encoder::new_centered(0.,10., 8, 0).unwrap();
//!
//!     // don't hesitate to print and checkout for yourself
//!     println!("{}",encoder);
//! }
//! ```
//!
//! We can now use one of these [Encoders](super::super::crypto_api::Encoder) to properly **encode some messages** as follow.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // create an Encoder instance where messages are in the interval [-10, 10] with 8 bits of precision
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // create a list of messages in our interval
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // create a new Plaintext instance filled with the plaintexts we want
//!     let plaintext = Plaintext::encode(&messages, &encoder);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! As you can see, we have **a list of 5 floating numbers** in our interval (our messages), and we are encoding them inside **an instance of [Plaintext](super::super::crypto_api::Plaintext)** thanks to our **[Encoder](super::super::crypto_api::Encoder)**.
//!
//! Note that we could have created 5 different instances of Plaintext to store one plaintext at a time.
//!  
//!
//! # Step 3: Let's encrypt plaintexts with LWE ciphertexts
//!
//! LWE stands for [Learning With Errors](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf) and refers to a **computational problem** conjectured to be **hard to solve**.
//! An VectorLWE ciphertext is composed of a **vector of integers called ``mask``** and an **integer called ``body``**.
//! The size of the mask vector is called ``dimension`` and some noise is added into the ``body`` value.
//! This noise is taken from a **normal distribution** centered in zero with a specified standard
//! deviation.
//! The **security of an LWE ciphertext** depends on this dimension and this standard deviation.
//! The security also depends on the way the secret key is generated, and in the Concrete library there are only **uniformly random binary secret keys** for now.
//! Note that with LWE ciphertext, every computation is performed in a modular integer ring such as Z/qZ where q is the modulus.
//!
//! ## 3-A. Pick your security parameters
//!
//! As mentioned above, some parameters are directly related to the **security of the
//! cryptosystem** such as the dimension of the LWE mask or the standard deviation of the noise
//! distribution.
//! This is why there is a structure called [LWEParams](super::super::crypto_api::LWEParams) that stores those parameters.
//! There are also a few sets of parameters for LWE ciphertexts which were estimated with the [LWE estimator](https://bitbucket.org/malb/lwe-estimator/src/master/) on September 15th 2020.
//!
//! ```rust
//! use concrete::*;
//!
//! // 128 bits of security:
//! // - with a dimension of 630
//! let lwe128_630 = LWEParams::new(630, -14);
//! // - with a dimension of 650
//! let lwe128_650 = LWEParams::new(650, -15);
//! // - with a dimension of 690
//! let lwe128_688 = LWEParams::new(690, -16);
//!
//! // 80 bits of security:
//! // - with a dimension of 630
//! let lwe80_630 = LWEParams::new(630, -24);
//! // - with a dimension of 650
//! let lwe80_650 = LWEParams::new(650, -25);
//! // - with a dimension of 690
//! let lwe80_688 = LWEParams::new(690, -26);
//! ```
//!
//! As you can see, there is a **trade-off** between the dimension and the standard deviation: the **larger the dimension** is, the **smaller the standard deviation** has to be to keep the same level of security.
//! This trade-off has some consequences.
//! The bigger the dimension, the more computation has to be done for each cryptographic operation (**slow down**), but also the bigger are every ciphertexts (**size overhead**).
//! However, the smaller the standard deviation is, the smaller the noise is, the **more bits** we have for the **precision** of the messages!
//! There are other consequences, and we will talk about them during the next parts of this guide.
//! We emphasize how **important** this choice of parameters is for any practical use case.
//!
//! There are also **more LWE sets of parameters** defined as constants in the [lwe_params](super::super::crypto_api::lwe_params) module, so we can call an LWE set of parameters with 128 bits of security and a mask size of 630 like that:
//!
//! ```rust
//! use concrete::*;
//!
//! fn main() {
//!     let lwe_params = LWE128_630;
//!
//!     // don't hesitate to print and checkout for yourself
//!     println!("{}", lwe_params);
//! }
//! ```
//!
//! With an [LWEParams](super::super::crypto_api::LWEParams), it is now possible to generate a secret key.
//!
//! ## 3-B. Generate your first LWE secret key
//!
//! In order to encrypt plaintexts, we need to generate a **secret key**.
//! In the Concrete library, an LWE secret key is represented as an **[LWESecretKey](super::super::crypto_api::LWESecretKey) structure**, and is binary (only composed of zeros and ones) and uniformly picked at random.
//! To construct a new [LWESecretKey](super::super::crypto_api::LWESecretKey) instance, we simply call the [new](super::super::crypto_api::LWESecretKey::new) function that takes as input an [LWEParams](super::super::crypto_api::LWEParams).
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // settle for a set of LWE parameters
//!     let lwe_params = LWE128_630;
//!
//!     // generate a fresh secret key
//!     let secret_key = LWESecretKey::new(&lwe_params);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 3-C. And finally encrypt your messages
//!
//! We now are able encrypt some plaintexts inside some **LWE ciphertexts**.
//! We simply have to use the [VectorLWE](super::super::crypto_api::VectorLWE) struct and its [encrypt](super::super::crypto_api::VectorLWE::encrypt) function.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_630);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
//!     let ciphertext = VectorLWE::encrypt(&secret_key, &plaintext);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Congratulations!
//! You just encoded your messages into plaintexts and then encrypted those into ciphertexts.
//! Just so you know, there is also a **quicker** way to do the encryption without having to define a [Plaintext](super::super::crypto_api::Plaintext).
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_630);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 3-D. Decrypt and decode your messages
//!
//! In this part of the guide we do not compute homomorphic operations yet.
//! It is the purpose of the next parts.
//! Now we have to learn how to **decrypt**.
//! The decryption is split into **2 consecutive steps**:
//! - the **phase computation**, which cleans the ciphertext from the secret key
//! - the **decoding**, which removes the noise and go back to a proper message.
//!
//! You can execute those 2 steps in one call of the [decrypt](super::super::crypto_api::VectorLWE::decrypt) function as follow:
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 6, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = LWESecretKey::new(&LWE128_630);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
//!     let ciphertext = VectorLWE::encrypt(&secret_key, &plaintext).unwrap();
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before, after) in messages.iter().zip(decryptions.iter()) {
//!         if (before - after).abs() > encoder.get_granularity() / 2. {
//!             panic!(
//!                 "before {} / after {} / half_gran {}",
//!                 before,
//!                 after,
//!                 encoder.get_granularity() / 2.
//!             );
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! As you can see, we **checked** at the end that we only lost less precision than we specified in the Encoder.
//!
//! # Step 4: Let's encrypt messages with RLWE ciphertexts
//!
//! RLWE stands for [Ring Learning With Errors](https://eprint.iacr.org/2012/230.pdf) and refers to the extension of LWE over the ring of polynomials with coefficients in a modular integer ring.
//! It is also a **computational problem** conjectured to be **hard to solve**.
//! For security purpose the polynomial ring is generated with a [cyclotomic polynomial](https://en.wikipedia.org/wiki/Cyclotomic_polynomial), and to improve performances, one of the n-th cyclotomic polynomial is picked such that n is a power of two.
//! An RLWE ciphertext is composed of a **vector of polynomials called ``mask``** and a **polynomial called ``body``**.
//! The size of the mask vector is called ``dimension``.
//! The size of our polynomials is n (a power of two) and refers to the number of coefficients in any polynomial.
//! As in an LWE ciphertext, some noise is added into the ``body`` part (which is a polynomial), one small noise per coefficient.
//! Those noises are taken from a **normal distribution** centered in zero with a specified
//! standard deviation.
//! The **security of an RLWE ciphertext** depends on the dimension of the RLWE mask, the value n and the standard deviation for the noises.
//! The security also depends on the way the secret key is generated, and in the Concrete library there are only **uniformly random binary secret keys**, which means that an RLWE secret key is a vector of polynomials with binary coefficients.
//!
//! This time, an RLWE plaintext is a **polynomial** of a certain degree.
//! It means that we can have **as many LWE plaintexts**  in an RLWE plaintext, that there are coefficients in the polynomial.
//! Each monomial of the RLWE plaintext polynomial is exactly like one LWE plaintext!
//! So when we will encrypt with RLWE, we will have the opportunity to **encrypt several messages at once** in a single RLWE ciphertext.
//!
//! ## 4-A. Pick your security parameters
//!
//! As mentioned above, some parameters are directly related to the **security of the
//! cryptosystem** such as the dimension of the RLWE mask, the number of coefficients in our
//! polynomials, or the standard deviation of the noise distribution.
//! This is why there is a structure called [RLWEParams](super::super::crypto_api::RLWEParams) that stores those parameters.
//! There is also a few sets of parameters for RLWE ciphertexts which were estimated with the [VectorLWE estimator](https://bitbucket.org/malb/lwe-estimator/src/master/) on September 15th 2020.
//! Note that the sizes of the polynomials are powers of two.
//!
//! ```rust
//! use concrete::*;
//!
//! // 128 bits of security:
//! // - with a polynomial size of 256 and a mask size of 4
//! let rlwe128_256_4 = RLWEParams::new(256, 4, -25);
//! // - with a polynomial size of 512 and a mask size of 1
//! let rlwe128_512_1 = RLWEParams::new(512, 1, -11);
//! // - with a polynomial size of 1024 and a mask size of 1
//! let rlwe128_1024_1 = RLWEParams::new(1024, 1, -25);
//!
//! // 80 bits of security:
//! // - with a polynomial size of 256 and a mask size of 2
//! let rlwe80_256_2 = RLWEParams::new(256, 2, -19);
//! // - with a polynomial size of 256 and a mask size of 1
//! let rlwe80_512_1 = RLWEParams::new(512, 1, -19);
//! // - with a polynomial size of 256 and a mask size of 4
//! let rlwe80_1024_1 = RLWEParams::new(1024, 1, -40);
//! ```
//!
//! There is still a **trade-off** but this time it is between the standard deviation and the product of polynomial size and the mask size: the **larger the product** is, the **smaller the standard deviation** has to be to keep the same level of security.
//! This trade-off has also some consequences.
//! The bigger the product, the more computation has to be done for each cryptographic operation (**slow down**), but also the bigger are every ciphertexts (**size overhead**).
//! However, the smaller the standard deviation is, the smaller the noise is, the **more bits** we have for the **precision** of the messages!
//! There are other consequences, and we will talk about them during the next parts of this guide.
//! We emphasize one more time how **important** this choice of parameters is for any practical use case.
//!
//! There is also **more RLWE sets of parameters** defined as constants in the [rlwe_params](super::super::crypto_api::rlwe_params) module, so we can call an RLWE set of parameters with 128 bits of security, a polynomial size set to 1024 and a mask size of 1 like that:
//!
//! ```rust
//! use concrete::*;
//!
//! fn main() {
//!     let rlwe_params = RLWE128_1024_1;
//!
//!     // don't hesitate to print and checkout for yourself
//!     println!("{}", rlwe_params);
//! }
//! ```
//!
//! With an [RLWEParams](super::super::crypto_api::RLWEParams), it is now possible to generate a secret key.
//!
//! ## 4-B. Generate your first RLWE secret key
//!
//! In order to encrypt plaintexts, we need to generate a **secret key**.
//! In the Concrete library, an RLWE secret key is represented as an **[RLWESecretKey](super::super::crypto_api::RLWESecretKey) structure**, and is binary (each coefficient of each polynomial of the key is either a zero or a one) and uniformly picked at random.
//! To construct a new [RLWESecretKey](super::super::crypto_api::RLWESecretKey) instance, we simply call the [new](super::super::crypto_api::RLWESecretKey::new) function that takes as input an [RLWEParams](super::super::crypto_api::RLWEParams).
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // settle for a set of RLWE parameters
//!     let rlwe_params = RLWE128_1024_1;
//!
//!     // generate a fresh secret key
//!     let secret_key = RLWESecretKey::new(&rlwe_params);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 4-C. And finally encrypt your messages
//!
//! We now are able encrypt some plaintexts inside an **RLWE ciphertext**.
//! We will start by filling the first 5 coefficients of an RLWE ciphertext (since we only have 5 messages).
//! We simply use the [VectorRLWE](super::super::crypto_api::VectorRLWE) struct and its [encrypt_packed](super::super::crypto_api::VectorRLWE::encrypt_packed) function.
//! This function takes as input the secret key, a [Plaintext](super::super::crypto_api::Plaintext) and an [RLWEParams](super::super::crypto_api::RLWEParams).
//! Note that there should be at most as many plaintexts in the Plaintext than the polynomial size (to fit in one RLWE ciphertext).
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // generate a fresh secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
//!     let ciphertext = VectorRLWE::encrypt_packed(&secret_key, &plaintext);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Congratulations!
//! You just encoded your messages into plaintexts and then encrypted those into a ciphertext.
//! Or you can use the [encode_encrypt_packed](super::super::crypto_api::VectorRLWE::encode_encrypt_packed) function that does both at the same time.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new_centered(105., 5., 8, 0).unwrap();
//!
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![106.65286348661301, 104.87845375069587, 105.46354804688922];
//!
//!     // encode and encrypt
//!     let mut ciphertext =
//!         VectorRLWE::encode_encrypt_packed(&secret_key, &messages, &encoder).unwrap();
//! }
//! ```
//!
//! ## 4-D. Decrypt and decode your messages
//!
//! In this part of the guide we do not compute homomorphic operations yet.
//! It is the purpose of the next parts.
//! Now we have to learn how to **decrypt** RLWE ciphertexts.
//! The decryption is still split into **2 consecutive steps**:
//! - the **phase computation**, which cleans the ciphertext from the secret key
//! - the **decoding**, which removes the noise and go back to a proper message.
//!
//! You can also execute those 2 steps in one call of the [decrypt](super::super::crypto_api::VectorRLWE::decrypt) function as follow:
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(-10., 10., 8, 0).unwrap();
//!
//!     // generate a fresh secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // a list of messages
//!     let messages: Vec<f64> = vec![-6.276, 4.3, 0.12, -1.1, 7.78];
//!
//!     // encode and encrypt
//!     let plaintext = Plaintext::encode(&messages, &encoder).unwrap();
//!     let ciphertext = VectorRLWE::encrypt_packed(&secret_key, &plaintext).unwrap();
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext.decrypt_decode(&secret_key).unwrap();
//!
//!     // check the precision loss related to the encryption
//!     for (before, after) in messages.iter().zip(decryptions.iter()) {
//!         if (before - after).abs() > encoder.get_granularity() / 2. {
//!             panic!();
//!         }
//!     }
//!     println!("Well done :-)");
//! }
//! ```
//!
//! This decrypt](super::super::crypto_api::VectorRLWE::decrypt) function **only decrypts** coefficients associated with a **valid encoder** and in the previous example, since we only had five messages, we only get five decryptions.
//! As you can see, we **checked** at the end that we only lost less precision than we specified in the Encoder.
//!
//! # Step 5: Read and write ciphertexts or secret keys in files
//!
//! Each structure can easily be saved into json files with the ``save`` method  and to read a json file and recover a struct already saved, you can use the ``load`` function.
//! A simple example with an RLWESecretKey shows how to do that.
//!
//! ```rust
//! /// file: main.rs
//! use concrete::*;
//!
//! fn main() {
//!     // generate a secret key
//!     let secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!
//!     // save the secret key
//!     secret_key.save("my_very_secret_key.json");
//!
//!     // ...
//!
//!     // load secret key
//!     let loaded_sk = RLWESecretKey::load("my_very_secret_key.json");
//! }
//! ```
//!
//! # Conclusion
//!
//! In this part, you learned:
//! - how to encode a message into a plaintext
//! - how instantiate cryptographic parameters for LWE and RLWE
//! - how to encrypt and decrypt with LWE ciphertexts
//! - how to encrypt and decrypt with RLWE ciphertexts
//! - how to read and write ciphertexts or secret keys in files
//!
//! Wow, that's already a lot!
//! You're all set for the [next part](super::guide_part3_simple_operations) where you will play with simple homomorphic operations ;-)
