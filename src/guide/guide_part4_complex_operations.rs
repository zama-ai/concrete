//! Part 4 of the Concrete guide where you learn how to compute complex homomorphic operations with LWE and RLWE ciphertexts.
//!
//! This guide assumes that your already know how compute simple homomorphic operation on LWE/RLWE ciphertexts with the Concrete library.
//! If not, [this part](super::guide_part3_simple_operations) is designed for you ;-)
//!
//! In this part we will compute **complex homomorphic operations**.
//! With "complex", we refer to homomoprhic operation that needs some public cryptographic material to be executed and also more complex than additions or multiplications by constants.
//!
//! # Step 1: The key switching operation
//!
//! It is possible to **convert an LWE/RLWE** ciphertext associated with a secret key into another LWE/RLWE ciphertext associated with **another secret key**.
//! It is called **key switch**.
//! To do so, we need **encryptions of the bits of the first secret key**, encrypted with the second secret key and it is called **key switching key**.
//! The other good thing is that we can at the same time **change dimension** of the LWE/RLWE mask, or the polynomial size of the RLWE ciphertext.
//!
//! ## 1-A. Generate a key switching key
//!
//! First we explain how to generate a **key switching key**.
//! We need to create two secret keys with the same [LWEParams](super::super::crypto_api::LWEParams) or two different.
//! Then we can call the [new](super::super::crypto_api::LWEKSK::new) method to get a new [LWEKSK](super::super::crypto_api::LWEKSK) structure.
//! This function takes as input what is called `base_log` and `level`.
//! Those two inputs will contribute to the **amount of noise** inside the output LWE ciphertext, to the **precision of the output message**, to the **computation cost** of the whole procedure and also the **size** of the key switching key.
//! Those trade-offs can be **tricky**, and it is important to find a **good balance**.
//! Generally, we try to keep a small `level` value.
//!
//! Here is an **example** with 3 `levels` and a `base_log` of 8.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // generate two secret keys
//!     let secret_key_before = LWESecretKey::new(&LWE128_1024);
//!     let secret_key_after = LWESecretKey::new(&LWE128_630);
//!
//!     // generate the key switching key
//!     let ksk = crypto_api::LWEKSK::new(&secret_key_before, &secret_key_after, 8, 3);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//!
//! ## 1-B. Key switch a ciphertext
//!
//! Now we can easily **key switch** some ciphertexts :-)
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoder
//!     let encoder = Encoder::new(100., 110., 8, 0).unwrap();
//!
//!     // generate two secret keys
//!     let secret_key_before = LWESecretKey::new(&LWE128_1024);
//!     let secret_key_after = LWESecretKey::new(&LWE128_630);
//!
//!     // generate the key switching key
//!     let ksk = crypto_api::LWEKSK::new(&secret_key_before, &secret_key_after, 8, 3);
//!
//!     // a list of messages that we encrypt
//!     let messages: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
//!     let mut ciphertext_before =
//!         VectorLWE::encode_encrypt(&secret_key_before, &messages, &encoder).unwrap();
//!
//!     // key switch
//!     let ciphertext_after = ciphertext_before.keyswitch(&ksk);
//!
//!     // decryption
//!     let decryptions: Vec<f64> = ciphertext_before.decrypt_decode(&secret_key_before).unwrap();
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
//! We used the [keyswitch](super::super::crypto_api::VectorLWE::keyswitch) method that output a new [VectorLWE](super::super::crypto_api::VectorLWE) structure.
//! There are **other functions** for the key switch such as [keyswitch](super::super::crypto_api::VectorLWE::keyswitch) that takes as input both the before and after [VectorLWE](super::super::crypto_api::VectorLWE) structures.
//!
//! We went from a dimension of 1024 to a dimension of 630!
//!
//! # Step 2: The bootstrapping operation
//!
//! It is possible to **convert an LWE** with a lot of noise into another LWE ciphertext encrypting the same message but containing **less noise**.
//! This is called **bootstrapping**.
//! To do so, we need a different kind of ciphertext (encrypting the secret key) that is called **bootstrapping key**.
//! We say "different" because it is not an LWE or RLWE ciphertext but a RGSW ciphertext.
//! RGSW ciphertexts are actually a sort of **collection of RLWE ciphertexts** encrypting only a **single polynomial**.
//!
//! The bootstrapping procedure is **very similar** to the key switch procedure.
//! The main two differences between them are that the bootstrap is **more costly** in terms of computation but it enables to **remove some noise!**
//!
//! Thanks to this procedure ([introduced by Gentry in 2009](https://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf)) that takes the noise down, we are able to **perform any arbitrary computation**.
//! This is precisely that ability that makes the cryptosystem **fully** homomorphic!
//!
//! ## 1-A. Generate a bootstrapping key
//!
//! The generation of the bootstrapping key is **very similar** to the generation of the key switching key.
//! Indeed, the bits of the first secret key have to be encrypted with the second secret key.
//! Those encryptions constitute the **bootstrapping key**.
//! Note that they are encrypted with **RGSW ciphertexts**.
//!
//! With a bootstrapping, it is also possible to **change the dimension** of the LWE mask and the **secret key**: we have as input an LWE ciphertext with a dimension d_before and a secret key sk_before, and we end up after bootstrapping it, with an LWE ciphertext (of the same message) with a dimension d_after and a secret key sk_after.
//! If we want to **go back** to the d_before dimension and the sk_before secret key, we can simply perform a **key switch** procedure right after the bootstrap.
//!
//! We can call the [new](super::super::crypto_api::LWEBSK::new) method to get a new [LWEBSK](super::super::crypto_api::LWEBSK) structure.
//! The [new](super::super::crypto_api::LWEBSK::new) function, generates a bootstrapping key, by taking as input `base_log` and `level` among others.
//! Those two inputs along with the mask size of the output LWE, will contribute to the **amount of noise** inside the output LWE ciphertext, to the **precision of the output message**, to the **computation cost** of the whole procedure and also the **size** of the bootstrapping key.
//! Those trade-offs can also be **tricky**, and it is important to find a **good balance**.
//! Generally, we try to keep a small `level` value, and an output LWE mask size greater or equal to 1024.
//! Note that the output LWE mask size is going to be a **power of two** because the bootstrapping procedure involves **RLWE ciphertexts**.
//! This is why, in order to generate a bootstrapping key, we need an **LWE secret key** and an **RLWE secret key**.
//! The output LWE sample will be decrypted with an LWE secret key **obtained from the RLWE secret key**.
//!
//! Here is an **example** with 3 `levels` and a `base_log` of 5.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // settings
//!     let base_log: usize = 5;
//!     let level: usize = 3;
//!
//!     // secret keys
//!     let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!     let secret_key_input = LWESecretKey::new(&LWE128_630);
//!     let secret_key_output = rlwe_secret_key.to_lwe_secret_key();
//!
//!     // bootstrapping key
//!     let bootstrapping_key = LWEBSK::new(&secret_key_input, &rlwe_secret_key, base_log, level);
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! ## 1-B. Bootstrap a ciphertext
//!
//! We are now able to compute a **bootstrap**, thanks to our [bootstrapping key](super::super::crypto_api::LWEBSK) and the [bootstrap_nth](super::super::crypto_api::VectorLWE::bootstrap_nth) method.
//! It output a new [VectorLWE](super::super::crypto_api::VectorLWE) instance with the result.
//!
//! It is **essential** to have at least **one bit of padding** in the input LWE plaintext.
//! It is related to the **cyclotomic ring** and the negative sign popping after multiplication.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoders
//!     let encoder_input = Encoder::new(-10., 10., 6, 1).unwrap();
//!
//!     // secret keys
//!     let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!     let secret_key_input = LWESecretKey::new(&LWE128_630);
//!     let secret_key_output = rlwe_secret_key.to_lwe_secret_key();
//!
//!     // bootstrapping key
//!     let bootstrapping_key = LWEBSK::new(&secret_key_input, &rlwe_secret_key, 5, 3);
//!
//!     // messages
//!     let message: Vec<f64> = vec![-5.];
//!
//!     // encode and encrypt
//!     let ciphertext_input =
//!         VectorLWE::encode_encrypt(&secret_key_input, &message, &encoder_input).unwrap();
//!
//!     // bootstrap
//!     let ciphertext_output = ciphertext_input
//!         .bootstrap_nth(&bootstrapping_key, 0)
//!         .unwrap();
//!
//!     // decrypt
//!     let decryption = ciphertext_output.decrypt_decode(&secret_key_output).unwrap();
//!
//!     if (decryption[0] - message[0]).abs() > encoder_input.get_granularity() {
//!         panic!(
//!             "decryption: {} / expected value: {} / granularity: {}",
//!             decryption[0],
//!             message[0],
//!             encoder_input.get_granularity()
//!         );
//!     }
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! # Step 3: Evaluate any function within a bootstrapping
//!
//! There are many ways to **represent an arbitrary function f**.
//! A simple way is to use a **look up table** which is a sort of dictionary with many couples (x,f(x)).
//!
//! The bootstrapping procedure starts by homomorphically decrypting the message m (in the input LWE), so that it produces a **different encryption of that message m**.
//! The next step uses that new encryption of m to implicitly **select f(m)** inside a provided **look up table**.
//! This selection produces as a result a **new LWE ciphertext** encrypting the value f(m).
//!
//! We want to emphasize that this bootstrapping procedure with too small parameters, **looses some precision** about the encrypted input message.
//! Once again the **choice of parameter** is very important to avoid this issue.
//!
//! It is in reality **more complex** than that, indeed the **encodings are to be considered** for the creation of the look up table.
//! This is actually a **good thing**, because it allows us to **change completely the encoding** of a message after a bootstrapping.
//! Above we described element of the look up table as couples (m,f(m)) but in reality, it is more like **(encode_before(m),encode_after(f(m)))** with encode_before the encoding function of the input, and encode_after the encoding function of the output.
//!
//! It is **essential** to have at least **one bit of padding** in the input LWE plaintext.
//!
//! We can perform a **bootstrap** that also evaluate a function over the message, thanks to our [bootstrapping key](super::super::crypto_api::LWEBSK) and the [bootstrap_nth_with_function](super::super::crypto_api::VectorLWE::bootstrap_nth_with_function) method.
//! We need to **provide a function** taking a ``f64`` as input and output a ``f64``.
//! The bootstrap output a new [VectorLWE](super::super::crypto_api::VectorLWE) instance with the result.
//!
//! In the following example, we encrypt the message ``-5`` with **one bit of padding**, we bootstrap it (the 0-th ciphertext in the structure) and evaluate the **square function**, so we end up with a LWE ciphertext of the message ``25``.
//! The input encoding works in the interval [-10,10] and since we evaluate the square function, the **output interval** has to be [0,100].
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoders
//!     let encoder_input = Encoder::new(-10., 10., 4, 1).unwrap();
//!     let encoder_output = Encoder::new(0., 100., 4, 0).unwrap();
//!
//!     // secret keys
//!     let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!     let secret_key_input = LWESecretKey::new(&LWE128_630);
//!     let secret_key_output = rlwe_secret_key.to_lwe_secret_key();
//!
//!     // bootstrapping key
//!     let bootstrapping_key = LWEBSK::new(&secret_key_input, &rlwe_secret_key, 5, 3);
//!
//!     // messages
//!     let message: Vec<f64> = vec![-5.];
//!
//!     // encode and encrypt
//!     let ciphertext_input =
//!         VectorLWE::encode_encrypt(&secret_key_input, &message, &encoder_input).unwrap();
//!
//!     // bootstrap
//!     let ciphertext_output = ciphertext_input
//!         .bootstrap_nth_with_function(&bootstrapping_key, |x| x * x, &encoder_output, 0)
//!         .unwrap();
//!
//!     // decrypt
//!     let decryption = ciphertext_output.decrypt_decode(&secret_key_output).unwrap();
//!
//!     if (decryption[0] - message[0] * message[0]).abs() > encoder_output.get_granularity() {
//!         panic!(
//!             "decryption: {} / expected value: {} / granularity: {}",
//!             decryption[0],
//!             message[0] * message[0],
//!             encoder_output.get_granularity()
//!         );
//!     }
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! As you can see we used a **lambda function** ``|x| x * x`` to describe the square function we apply on the message during the bootstrap.
//! It is important to observe that this procedure works well in case of **regular functions** however when it comes to less regular functions, we need to have a perfect precision and noise management to guarantee correctness.
//!
//! # Step 4: Multiply two LWE ciphertexts together with bootstrapping procedures
//!
//! One of the most exciting homomorphic operation is the **multiplication of two ciphertexts**.
//! We can compute this multiplication with two bootstrapping procedures.
//! It means that we have to generate a [bootstrapping key](super::super::crypto_api::LWEBSK).
//! Note that there are some constraints about the 2 different input encodings:
//! - they should work in **two intervals** with the **same size**
//! - they should both have the **same number of bits of padding**
//! - they should both have **at least 2 bits of padding**
//!
//! In the following example we will **multiply** the first ciphertext of our 2 [VectorLWE key](super::super::crypto_api::VectorLWE) structures and obtain a **new [VectorLWE key](super::super::crypto_api::VectorLWE) structure** filled with only one ciphertext which is the **expected result**.
//!
//! ```rust
//! use concrete::*;
//! /// file: main.rs
//! use itertools::izip;
//!
//! fn main() {
//!     // encoders
//!     let encoder_1 = Encoder::new(10., 20., 6, 2).unwrap();
//!     let encoder_2 = Encoder::new(-30., -20., 6, 2).unwrap();
//!
//!     // generate secret keys
//!     let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
//!     let secret_key_input = LWESecretKey::new(&LWE128_630);
//!     let secret_key_output = rlwe_secret_key.to_lwe_secret_key();
//!
//!     // bootstrapping key
//!     let bsk = LWEBSK::new(&secret_key_input, &rlwe_secret_key, 5, 3);
//!
//!     // two lists of messages
//!     let messages_1: Vec<f64> = vec![10.276, 14.3, 14.12, 11.1, 17.78];
//!     let messages_2: Vec<f64> = vec![-22., -27.5, -21.2, -29., -25.];
//!
//!     // encode and encrypt
//!     let ciphertext_1 =
//!         VectorLWE::encode_encrypt(&secret_key_input, &messages_1, &encoder_1).unwrap();
//!     let ciphertext_2 =
//!         VectorLWE::encode_encrypt(&secret_key_input, &messages_2, &encoder_2).unwrap();
//!
//!     // multiplication
//!     let ciphertext_res = ciphertext_1
//!         .mul_from_bootstrap_nth(&ciphertext_2, &bsk, 0, 0)
//!         .unwrap();
//!
//!     // decrypt
//!     let decryption = ciphertext_res.decrypt_decode(&secret_key_output).unwrap();
//!
//!     if (decryption[0] - messages_1[0] * messages_2[0]).abs()
//!         > ciphertext_res.encoders[0].get_granularity()
//!     {
//!         panic!(
//!             "decryption: {} / expected value: {}",
//!             decryption[0],
//!             messages_1[0] * messages_2[0]
//!         );
//!     }
//!
//!     println!("Well done :-)");
//! }
//! ```
//!
//! Also note that the resulting encoding have **automatically** changed to be able to contain any possible result.
//!
//! # Conclusion
//!
//! In this final part, you saw:
//! - how to change the key encrypting a ciphertext with a **key switch**
//! - how to reduce the noise inside a ciphertext with a **bootstrap**
//! - how to compute any function on a single ciphertext with a **bootstrap** and a **function**
//! - how to compute a **multiplication** between two ciphertexts
//!
//! Congratulations, you finished the Rust guide and you know now how to compute pretty much anything you want.
